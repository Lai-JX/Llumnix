# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import threading
import time
from typing import List, Tuple, Optional, Callable, Any

from llumnix.backends.utils import BarrierActor
from llumnix.backends.vllm.parallel_migration_backend import MigrationBackendResourcPool
import torch
from func_timeout import func_set_timeout
import ray
import ray.util.collective as col
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import ray.actor

from vllm.worker.cache_engine import CacheEngine

from llumnix.internal_config import MigrationConfig
from llumnix.backends.migration_backend_interface import MigrationBackendBase
from llumnix.logging.logger import init_logger
from llumnix.constants import (
    NUMPY_SUPPORTED_DTYPES_FOR_MIGRATION,
    RAYRPC_MIGRATION_TIMEOUT,
)
from llumnix.utils import random_uuid, ray_get_with_timeout
import numpy as np

logger = init_logger(__name__)


# Once worker died, proxy actor will not restart.
@ray.remote(num_cpus=0, max_concurrency=8, max_restarts=-1)
class ProxyActor:
    def __init__(self, is_driver_worker: bool, use_ray_spmd_worker: bool):
        self.is_driver_worker = is_driver_worker
        self.use_ray_spmd_worker = use_ray_spmd_worker

    def exec_method(self, handle: ray.actor.ActorHandle, from_driver_worker=None, *args, **kwargs) -> Any:
        if (from_driver_worker) is True or (self.is_driver_worker and not self.use_ray_spmd_worker):
            ret = ray_get_with_timeout(
                handle.execute_engine_method_async.remote(
                    "execute_driver_worker_method_async", *args, **kwargs
                )
            )
        else:
            ret = ray_get_with_timeout(
                handle.execute_method.options(concurrency_group="migate").remote(*args, **kwargs)
            )

        return ret


NUMPY_SUPPORTED_DTYPES = [torch.float32, torch.float16]


class RayRpcMigrationBackend(MigrationBackendBase):
    def __init__(self,
                 instance_id: str,
                 migration_config: MigrationConfig,
                 cache_engine: List[CacheEngine],
                 worker_rank: int,
                 worker_handle_list: List[ray.actor.ActorHandle],
                 local_rank: int,
                 scheduling_strategy: PlacementGroupSchedulingStrategy,
                 is_driver_worker: bool,
                 gpu_cache: Optional[List[List[torch.Tensor]]],
                 use_ray_spmd_worker: bool,
                 worker_stage_seq_group_metadata_callback: Callable) -> None:
        super().__init__()

        self.instance_id = instance_id
        self.migration_config = migration_config
        self.cache_engine = cache_engine

        self.worker_rank = worker_rank
        self.worker_handle_list = worker_handle_list
        self.local_rank = local_rank

        max_migration_concurrency = self.migration_config.max_migration_concurrency
        self.proxy_actor = ProxyActor.options(
            scheduling_strategy=scheduling_strategy,
            max_concurrency = max_migration_concurrency*2,
            name=f"ProxyActor_{self.instance_id}_{random_uuid()}").remote(
                is_driver_worker, use_ray_spmd_worker
            )

        if self.cache_engine[0].dtype in NUMPY_SUPPORTED_DTYPES_FOR_MIGRATION:
            self.rpc_dtype = self.cache_engine[0].dtype
        else:
            self.rpc_dtype = torch.float32
            logger.warning("Detect numpy unsupported dtype: {}. Using torch.float32.".format(self.cache_engine[0].dtype))

        self.gpu_cache = gpu_cache
        self.use_ray_spmd_worker = use_ray_spmd_worker
        self.worker_stage_seq_group_metadata_callback = worker_stage_seq_group_metadata_callback

        self.cache_device = "cpu"
        self.num_migration_buffer_blocks = self.migration_config.migration_buffer_blocks
        self.num_layers = self.cache_engine[0].num_attention_layers
        self.migration_cache_size = self.cache_engine[0].block_size * self.cache_engine[0].num_kv_heads * self.cache_engine[0].head_size

        self.buffer_pool = MigrationBackendResourcPool((self.num_migration_buffer_blocks, self.num_layers, 2, self.migration_cache_size),
                                                       self.cache_engine[0].dtype,
                                                       self.cache_device,
                                                       self.local_rank,
                                                       max_migration_concurrency,)
        logger.info(f"buffer size {max_migration_concurrency} * {(self.num_migration_buffer_blocks, self.num_layers, 2, self.cache_engine[0].block_size, self.cache_engine[0].num_kv_heads, self.cache_engine[0].head_size)} * {self.cache_engine[0].dtype}")
        self.request_id_buffer = {}
        # self.dummy_cache = torch.empty(
        #     size=(self.num_migration_buffer_blocks, self.num_layers, 2, self.migration_cache_size),
        #     dtype=self.cache_engine[0].dtype,
        #     device=self.cache_device,
        #     pin_memory=True
        # )
        # self.migration_stream = torch.cuda.Stream()
        # self.send_cache_split = None
        # self.barrier_actor = None
        # self.wait_for_split_event = threading.Event()

    def init_backend(self, group_name: str, world_size: int, rank: int) -> bool:
        logger.info("Create rayrpc migration backend successfully.")
        return True

    def destory_backend(self) -> None:
        # rpc migration backend does not need to be destroyed as there is no group.
        # It uses ray actor handle to migration cache blocks.
        pass

    def warmup(self) -> bool:
        ray_get_with_timeout(
            self.proxy_actor.exec_method.remote(
                self.worker_handle_list[self.worker_rank], "do_send", None, [0]
            )
        )
        logger.info("Rayrpc migration backend warmup successfully.")
        return True

    # The src actor will pack the kv-cache data layer by layer. Specifically, NumPy is used for the transfer
    # because, for a single node, Ray RPC can transfer NumPy arrays via shared memory. Then, the recv actor
    # first copies the data to a pinned-memory dummy cache before transferring it to the GPU to accelerate data transfer.
    def recv_cache(self,
                   request_id: str,
                   src_worker_handle: ray.actor.ActorHandle,
                   src_blocks: List[int],
                   dst_blocks: List[int],
                   is_last_stage: bool,
                      chunk_size: int=1,
                      chunk_rank: int=0) -> None:
        tot_blocks = len(src_blocks)
        from_driver_worker = (self.worker_rank // chunk_size) == 0
        rpc_numpy_cache = None
        src_seq_group_metadata = None
        logger.info("migrate_cache , chunk_rank: {}, worker_rank:{}, tot_blocks: {}, chunk_size: {}"
                    .format( chunk_rank, self.worker_rank, tot_blocks, chunk_size))
        # if chunk_rank == 0:
        #     logger.info("migrate_cache , chunk_rank == 0")
        #     self.barrier_actor = BarrierActor.options().remote(chunk_size) 
        for start_idx in range(0, tot_blocks, self.num_migration_buffer_blocks):
            offset = min(self.num_migration_buffer_blocks, tot_blocks - start_idx)
            is_last_comm = (tot_blocks - start_idx <= self.num_migration_buffer_blocks)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            send_worker_metadata = self.use_ray_spmd_worker and is_last_stage and is_last_comm
            # TODO(s5u13b): Remote call has serialization cost, optimize it.
            ray_obj = self.proxy_actor.exec_method.remote(
                src_worker_handle,
                from_driver_worker,
                "do_send",
                None,
                send_blocks,
                request_id=request_id,
                send_worker_metadata=send_worker_metadata,
                chunk_size=chunk_size, chunk_rank=chunk_rank
            )
            if rpc_numpy_cache is not None:
                self.do_recv(request_id, rpc_numpy_cache, recv_blocks)
            recv_blocks = dst_blocks[start_idx:start_idx+offset]

            if send_worker_metadata:
                rpc_numpy_cache, src_seq_group_metadata = ray_get_with_timeout(ray_obj, timeout=RAYRPC_MIGRATION_TIMEOUT)
            else:
                rpc_numpy_cache = ray_get_with_timeout(ray_obj)

        self.do_recv(request_id, rpc_numpy_cache, recv_blocks)
        if src_seq_group_metadata:
            self.worker_stage_seq_group_metadata_callback(request_id, src_seq_group_metadata)
        # ray.get(self.barrier_actor.arrive.remote())

    def migrate_cache_subtract_tp(self,
                                  request_id: str,
                      src_handle: List["ray.actor.ActorHandle"],
                      src_blocks: List[int],
                      dst_blocks: List[int],
                      is_last_stage: bool,
                      chunk_size: int=1) -> None:
        tot_blocks = len(src_blocks)
        rpc_numpy_cache = None
        src_seq_group_metadata = None
        logger.info("migrate_cache  worker_rank:{}, tot_blocks: {}, chunk_size: {}"
                    .format(self.worker_rank, tot_blocks, chunk_size))

        for start_idx in range(0, tot_blocks, self.num_migration_buffer_blocks):
            offset = min(self.num_migration_buffer_blocks, tot_blocks - start_idx)
            is_last_comm = (tot_blocks - start_idx <= self.num_migration_buffer_blocks)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            send_worker_metadata = self.use_ray_spmd_worker and is_last_stage and is_last_comm
            tasks = []
            for idx, handle in enumerate(src_handle):
                from_driver_worker = (idx == 0 and self.worker_rank == 0)
                tasks.append(
                    self.actor.exec_method.remote(handle, from_driver_worker, "do_send",
                    None, send_blocks, request_id=request_id, send_worker_metadata=send_worker_metadata)
                )
            ray_objs = ray.get(tasks)

            
            if rpc_numpy_cache is not None:
                self.do_recv(rpc_numpy_cache, recv_blocks)
            recv_blocks = dst_blocks[start_idx:start_idx+offset]

            if send_worker_metadata:
                rpc_numpy_cache, src_seq_group_metadata = ray_objs[:,0], ray_objs[:,1]
            else:
                rpc_numpy_cache = ray_objs
            logger.info("migrate_cache_subtract_tp, chunk_size: {}, start_idx: {}, offset: {}, rpc_numpy_cache shape: {}"
                        .format(chunk_size, start_idx, offset, rpc_numpy_cache[0].shape))
            ss_time = time.time()
            for i in range(len(rpc_numpy_cache)):
                rpc_numpy_cache[i] = rpc_numpy_cache[i].reshape(
                    self.num_layers, 2, len(send_blocks),
                    self.cache_engine[0].block_size,
                    self.cache_engine[0].num_kv_heads // chunk_size,
                    self.cache_engine[0].head_size
                )
            rpc_numpy_cache = np.concatenate(rpc_numpy_cache, axis=4)
            rpc_numpy_cache = rpc_numpy_cache.reshape( self.num_layers, 2, len(send_blocks), self.migration_cache_size)
            logger.info("migrate_cache_subtract_tp, concatenate and reshape cost: {}"
                        .format(time.time()-ss_time))
        self.do_recv(rpc_numpy_cache, recv_blocks)
        if src_seq_group_metadata:
            self.worker_stage_seq_group_metadata_callback(request_id, src_seq_group_metadata)

    def do_send(self, request_id, dst_handle: "ray.actor.ActorHandle", blocks: List[int], virtuel_engine: int=0, chunk_size=1, chunk_rank=0):
        num_blocks = len(blocks)
        ss = time.time()
        if chunk_rank == 0:
            ss = time.time()
            buffer = self.buffer_pool.acquire()
            self.request_id_buffer[request_id] = buffer
            if chunk_size > 1:
                # self.barrier_actor = BarrierActor.options().remote(chunk_size)
                buffer.barrier = threading.Barrier(chunk_size)
            send_cache = buffer.dummy_cache[:num_blocks].view(self.num_layers, 2, num_blocks, self.migration_cache_size)
            # send_cache = self.dummy_cache[:num_blocks].view(self.num_layers, 2, num_blocks, self.migration_cache_size)
            # src_to_dst = {block_num: idx for idx, block_num in enumerate(blocks)}
            src_to_dst: List[Tuple[int, int]] = []
            for idx in range(num_blocks):
                src_to_dst.append((blocks[idx], idx))
            block_mapping_tensor = torch.tensor(src_to_dst,
                                                dtype=torch.int64,
                                                device="cpu", pin_memory=True).view(-1, 2)
            logger.info(f"time[do_send][{chunk_rank}] before swap_blocks : {time.time()-ss}")
        if chunk_rank == 0:
            with torch.cuda.stream(buffer.migration_stream):
                for layer_idx in range(self.num_layers):
                    self.cache_engine[virtuel_engine].attn_backend \
                        .swap_blocks(self.gpu_cache[virtuel_engine][layer_idx], send_cache[layer_idx], block_mapping_tensor)
            torch.cuda.Stream.synchronize(buffer.migration_stream)
            if chunk_size == 1:
                ret = send_cache.to(self.rpc_dtype).numpy()
                del self.request_id_buffer[request_id]
                self.buffer_pool.release(buffer)
                return ret
            logger.info(f"time[do_send][{chunk_rank}] after swap_blocks : {time.time()-ss}")
            # logger.info("shape before split: {}".format(send_cache.shape))
            send_cache = send_cache.view(
                self.num_layers, 2, num_blocks,
                self.cache_engine[0].block_size,
                self.cache_engine[0].num_kv_heads,
                self.cache_engine[0].head_size
            )
            logger.info(f"time[do_send][{chunk_rank}] after view : {time.time()-ss}")
            # 按照num_kv_heads所在维度进行划分
            buffer.send_cache_split = list(torch.chunk(send_cache, chunk_size, dim=4))
            logger.info(f"time[do_send][{chunk_rank}] after split : {time.time()-ss}")
            # logger.info("shape after split: {} + {}; {}".format(self.send_cache_split[0].shape,self.send_cache_split[1].shape,self.migration_cache_size // chunk_size))
            if chunk_size > 1:
                buffer.wait_for_split_event.set()
        else:
            # 等待划分完成
            while True:
                # 根据requst_id获取buffer
                buffer = self.request_id_buffer.get(request_id)
                logger.info(f'can not find buffer for {request_id}')
                if buffer is not None:
                    break
            buffer.wait_for_split_event.wait()
        logger.info(f"time[do_send][{chunk_rank}] split finished and all process begin : {time.time()-ss}")
        
        # logger.info("shape after split[{}]: {} + {}; {}".format(chunk_rank,self.send_cache_split[0].shape,self.send_cache_split[1].shape,self.migration_cache_size // chunk_size))
        buffer.send_cache_split[chunk_rank] = buffer.send_cache_split[chunk_rank].reshape(
            self.num_layers, 2, num_blocks, self.migration_cache_size // chunk_size
        )
        logger.info(f"time[do_send][{chunk_rank}] process reshape : {time.time()-ss}")
        # ray.get(self.barrier_actor.arrive.remote())
        buffer.barrier.wait()
        logger.info(f"time[do_send][{chunk_rank}] after barrier_actor : {time.time()-ss}")
        if chunk_rank == 0 and chunk_size > 1:
            buffer.wait_for_split_event.clear()
            self.buffer_pool.release(buffer)
            del self.request_id_buffer[request_id]
        return buffer.send_cache_split[chunk_rank].to(self.rpc_dtype).numpy()

    # pylint: disable=arguments-differ
    def do_recv(self, request_id, src_worker_handle: ray.actor.ActorHandle, blocks: List[int], virtuel_engine: int=0) -> None:
        num_blocks = len(blocks)
        # src_to_dst = dict(enumerate(blocks))
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((idx, blocks[idx]))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu", pin_memory=True).view(-1, 2)
        buffer = self.buffer_pool.acquire()
        self.request_id_buffer[request_id] = buffer
        recv_cache = buffer.dummy_cache[:num_blocks].view(self.num_layers, 2, num_blocks, self.migration_cache_size)
        # use pin memory dummy_cache to speed up data transfer
        recv_cache.copy_(torch.from_numpy(src_worker_handle))

        with torch.cuda.stream(buffer.migration_stream):
            for layer_idx in range(self.num_layers):
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(recv_cache[layer_idx], self.gpu_cache[virtuel_engine][layer_idx], block_mapping_tensor)
        torch.cuda.Stream.synchronize(buffer.migration_stream)
        del self.request_id_buffer[request_id]
        self.buffer_pool.release(buffer)


def try_import_gloo():
    try:
        # pylint: disable=C0415
        from ray.util.collective.collective_group import gloo_util
        import pygloo

        # Add support for bf16 type in Gloo. Now bf16 and fp16 both map to glooFloat16, but this is okay because
        # Gloo only uses the data type size for transmission.
        # pylint: disable=W0221,I1101
        gloo_util.TORCH_GLOO_DTYPE_MAP[torch.bfloat16] = pygloo.glooDataType_t.glooFloat16
    except ImportError as e:
        raise ImportError("Gloo is not installed. Please install it first.") from e


class RayColMigrationBackend(MigrationBackendBase):
    def __init__(self,
                 instance_id: str,
                 migration_config: MigrationConfig,
                 cache_engine: List[CacheEngine],
                 local_rank: int,
                 worker_rank: int,
                 scheduling_strategy: PlacementGroupSchedulingStrategy,
                 is_driver_worker: bool,
                 gpu_cache: Optional[List[List[torch.Tensor]]],
                 use_ray_spmd_worker: bool,
                 worker_stage_seq_group_metadata_callback: Callable,
                 ) -> None:
        super().__init__()

        # pylint: disable=C0415
        import cupy

        self.instance_id = instance_id
        self.migration_config = migration_config
        self.cache_engine = cache_engine
        self.backend = migration_config.migration_backend
        # migration_config.migration_num_layers 默认是1
        self.migration_num_layers = min(migration_config.migration_num_layers, self.cache_engine[0].num_attention_layers)
        self.num_migration_buffer_blocks = migration_config.migration_buffer_blocks

        self.backend = migration_config.migration_backend
        self.global_world_size = -1
        self.global_rank = -1
        self.group_name = None

        self.local_rank = local_rank
        self.worker_rank = worker_rank
        self.proxy_actor = ProxyActor.options(
            scheduling_strategy=scheduling_strategy,
            name=f"ProxyActor_{self.instance_id}_{random_uuid()}").remote(
                is_driver_worker, use_ray_spmd_worker
            )
        self.gpu_cache = gpu_cache
        self.use_ray_spmd_worker = use_ray_spmd_worker
        self.worker_stage_seq_group_metadata_callback = worker_stage_seq_group_metadata_callback

        self.migration_cache_size = self.cache_engine[0].block_size * self.cache_engine[0].num_kv_heads * self.cache_engine[0].head_size

        if self.backend == 'gloo':
            try_import_gloo()
            self.cache_device = "cpu"
        else:
            self.cache_device = torch.device(f"cuda:{self.local_rank}")

        pin_memory = (self.backend == 'gloo')
        self.dummy_cache = torch.empty(
            size=(self.num_migration_buffer_blocks, self.migration_num_layers, 2, self.migration_cache_size),
            dtype=self.cache_engine[0].dtype,
            device=self.cache_device,
            pin_memory=pin_memory
        )
        with cupy.cuda.Device(self.local_rank):
            self.migration_stream = cupy.cuda.Stream()
        self.send_cache_split = None
        self.barrier_actor = None   # BarrierActor for chunk migration
        self.wait_for_split_event = threading.Event()

    def init_backend(self, group_name: str, world_size: int, rank: int) -> bool:
        @func_set_timeout(self.migration_config.migration_backend_init_timeout)
        def init_group(world_size, rank, backend, group_name):
            col.init_collective_group(world_size, rank, backend, group_name)

        self.group_name = group_name
        self.global_world_size = world_size
        self.global_rank = rank

        try:
            init_group(world_size, rank, self.backend, group_name)
        # pylint: disable=broad-except
        except Exception:
            self._log_exception("init_backend")
            return False

        self._log_success("init_backend")
        return True

    def destory_backend(self) -> None:
        if self.group_name is None:
            return

        err_info = None
        try:
            col.destroy_collective_group(self.group_name)
        # pylint: disable=W0703
        except Exception as e:
            err_info = e

        if err_info is not None:
            self._log_exception("destory_backend")
        else:
            self._log_success("destory_backend")

        self.group_name = None

    def warmup(self) -> bool:
        if self.global_world_size > 1:
            try:
                # 用 shape=(1,) 的张量做 allreduce，保证所有进程 shape 一致
                test_tensor = torch.tensor([1.0], dtype=self.cache_engine[0].dtype, device=self.cache_device)
                col.allreduce(test_tensor, self.group_name)
                # col.allreduce(self.dummy_cache[0], self.group_name)
            # pylint: disable=W0703
            except Exception:
                self._log_exception("warmup")
                return False
        self._log_success("warmup")
        return True

    def _log_success(self, func_name: str):
        logger.info(
            "Migration backend {} success "
            "(group_name: {}, world_size: {}, rank: {}, backbend: {})".format(
                func_name, self.group_name, self.global_world_size, self.global_rank, self.backend
            )
        )

    def _log_exception(self, func_name: str):
        logger.exception(
            "Error in migration backend {} "
            "(group_name: {}, world_size: {}, rank: {}, backbend: {})".format(
                func_name, self.group_name, self.global_world_size, self.global_rank, self.backend
            )
        )

    # Ray.collective is used to construct the gloo and nccl backends. The do_send/do_recv functions will transmit
    # data layer by layer. Take into consideration that col.send/recv are blocking operations.
    def recv_cache(self,
                   request_id: str,
                   src_worker_handle: ray.actor.ActorHandle,
                   src_blocks: List[int],
                   dst_blocks: List[int],
                   is_last_stage: bool,
                      chunk_size: int=1,
                      chunk_rank: int=0) -> None:
        tot_blocks = len(src_blocks)
        from_driver_worker = (self.worker_rank // chunk_size) == 0
        src_rank = ray_get_with_timeout(self.proxy_actor.exec_method.remote(src_worker_handle, from_driver_worker, "get_global_rank"))

        src_seq_group_metadata = None
        for start_idx in range(0, tot_blocks, self.num_migration_buffer_blocks):
            offset = min(self.num_migration_buffer_blocks, tot_blocks - start_idx)
            is_last_comm = (tot_blocks - start_idx <= self.num_migration_buffer_blocks)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            recv_blocks = dst_blocks[start_idx:start_idx+offset]
            send_worker_metadata = self.use_ray_spmd_worker and is_last_stage and is_last_comm
            ray_obj = self.proxy_actor.exec_method.remote(
                src_worker_handle,
                "do_send",
                self.global_rank,
                send_blocks,
                request_id=request_id,
                send_worker_metadata=send_worker_metadata,
                chunk_size=chunk_size, chunk_rank=chunk_rank
            )
            # Ray collective communication does not have timeout parameters,
            # and run this method in another thread to set timeout will also cause cuda stream device mismatch error,
            # so recv cache does not have timeout only when the migration backend is ray collective.
            self.do_recv(src_rank, recv_blocks)
            if send_worker_metadata:
                _, src_seq_group_metadata = ray_get_with_timeout(ray_obj)
        if src_seq_group_metadata:
            self.worker_stage_seq_group_metadata_callback(request_id, src_seq_group_metadata)

    def migrate_cache_subtract_tp(self,
                      src_handle: List["ray.actor.ActorHandle"],
                      src_blocks: List[int],
                      dst_blocks: List[int],
                      request_id: str,
                      is_last_stage: bool,
                      chunk_size: int=1) -> None:
        tot_blocks = len(src_blocks)
        from_driver_worker = (self.worker_rank // chunk_size) == 0
        tasks = []
        ss = time.time()
        for idx, handle in enumerate(src_handle):
            from_driver_worker = (idx == 0 and self.worker_rank == 0)
            tasks.append(
                self.actor.exec_method.remote(handle, from_driver_worker, "get_global_rank")
            )
        src_ranks = ray.get(tasks)
        logger.info(f"time[do_send] after src_ranks : {time.time()-ss}")
        src_seq_group_metadata = None
        for start_idx in range(0, tot_blocks, self.num_migration_buffer_blocks):
            offset = min(self.num_migration_buffer_blocks, tot_blocks - start_idx)
            is_last_comm = (tot_blocks - start_idx <= self.num_migration_buffer_blocks)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            recv_blocks = dst_blocks[start_idx:start_idx+offset]
            send_worker_metadata = self.use_ray_spmd_worker and is_last_stage and is_last_comm
            tasks = []
            for idx, handle in enumerate(src_handle):
                from_driver_worker = (idx == 0 and self.worker_rank == 0)
                tasks.append(
                    self.actor.exec_method.remote(handle, from_driver_worker, "do_send",
                    self.global_rank, send_blocks, request_id=request_id, send_worker_metadata=send_worker_metadata)
                )
            
            logger.info(f"time[do_send] before do_recv : {time.time()-ss}")
            self.do_recv(src_ranks, recv_blocks, 0, chunk_size)
            logger.info(f"time[do_send] after do_recv : {time.time()-ss}")
            if send_worker_metadata:
                ray_objs = ray.get(tasks)
                _, src_seq_group_metadata = ray_objs[:,0], ray_objs[:,1]
        if src_seq_group_metadata:
            self.worker_stage_seq_group_metadata_callback(request_id, src_seq_group_metadata)

    def do_send(self, dst_worker_handle: "ray.actor.ActorHandle", blocks: List[int], virtuel_engine: int=0, chunk_size=1, chunk_rank=0):
        import cupy
        num_blocks = len(blocks)
        # logger.info("do_send: {} -> {}, chunk_rank: {}, worker_rank:{}, local_rank:{}, num_blocks: {}"
        #             .format(self.global_rank, dst_handle, chunk_rank, self.worker_rank, self.local_rank,num_blocks))
        if chunk_rank == 0:
            # self.barrier_actor = BarrierActor.options().remote(chunk_size)
            if chunk_size > 1:
                self.barrier = threading.Barrier(chunk_size)
            send_cache = self.dummy_cache[:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
            src_to_dst: List[Tuple[int, int]] = []
            for idx in range(num_blocks):
                src_to_dst.append((blocks[idx], idx))
            block_mapping_tensor = torch.tensor(src_to_dst,
                                                dtype=torch.int64,
                                                device="cpu", pin_memory=True).view(-1, 2)
        with cupy.cuda.Device(self.local_rank):
            with self.migration_stream:
                for layer_idx in range(self.cache_engine[0].num_attention_layers):
                    cache_idx = layer_idx % self.migration_num_layers
                    if chunk_rank == 0:
                        self.cache_engine[virtuel_engine].attn_backend \
                            .swap_blocks(self.gpu_cache[virtuel_engine][layer_idx], send_cache[cache_idx], block_mapping_tensor)
                        
                    if cache_idx + 1 == self.migration_num_layers or layer_idx + 1 == self.cache_engine[0].num_attention_layers:
                        if chunk_size == 1:
                            col.send(send_cache, dst_worker_handle, self.group_name)
                        # logger.info("do_send: {} -> {}, layer_idx: {}, chunk_rank: {}"
                                    # .format(self.global_rank, dst_handle, layer_idx, chunk_rank,))
                        else:
                            ss_time = time.time()
                            if chunk_rank == 0:
                                # logger.info("shape before split: {}".format(send_cache.shape))
                                send_cache = send_cache.view(
                                    self.migration_num_layers, 2, num_blocks,
                                    self.cache_engine[0].block_size,
                                    self.cache_engine[0].num_kv_heads,
                                    self.cache_engine[0].head_size
                                )
                                # 按照num_kv_heads所在维度进行划分
                                self.send_cache_split = list(torch.chunk(send_cache, chunk_size, dim=4))
                                # logger.info("shape after split: {} + {}; {}".format(self.send_cache_split[0].shape,self.send_cache_split[1].shape,self.migration_cache_size // chunk_size))
                                if chunk_size > 1:
                                    self.wait_for_split_event.set()
                            else:
                                # 等待划分完成
                                self.wait_for_split_event.wait()
                            
                            # logger.info("shape after split[{}]: {} + {}; {}".format(chunk_rank,self.send_cache_split[0].shape,self.send_cache_split[1].shape,self.migration_cache_size // chunk_size))
                            self.send_cache_split[chunk_rank] = self.send_cache_split[chunk_rank].reshape(
                                self.migration_num_layers, 2, num_blocks, self.migration_cache_size // chunk_size
                            )
                            
                            # TODO(KuilongCui): check the error code if peer is dead
                            col.send(self.send_cache_split[chunk_rank], dst_worker_handle, self.group_name)
                            # logger.info("do_send finished: {} -> {}, layer_idx: {}, chunk_rank: {}, cost: {}"
                                        # .format(self.global_rank, dst_handle, layer_idx, chunk_rank, time.time()-ss_time))
                            # 等待各个进程都传输完成
                            # ray.get(self.barrier_actor.arrive.remote())
                            self.barrier.wait()
                            if chunk_size > 1 and chunk_rank == 0:
                                self.wait_for_split_event.clear()
                            # logger.info("do_send all finished: {} -> {}, layer_idx: {}, chunk_rank: {}, cost: {}"
                            #             .format(self.global_rank, dst_handle, layer_idx, chunk_rank, time.time()-ss_time))
                        
                
            self.migration_stream.synchronize()
        # logger.info("do_send finished: {} -> {}, chunk_rank: {}"
        #             .format(self.global_rank, dst_handle, chunk_rank,))

    def do_recv(self, src_worker_handle: ray.actor.ActorHandle, blocks: List[int], virtuel_engine: int=0, chunk_size=1) -> None:
        def recv_worker(idx, group_rank):
            col.recv(self.send_cache_split[idx], group_rank, self.group_name)
            self.send_cache_split[idx] = self.send_cache_split[idx].reshape(
                                self.migration_num_layers, 2, num_blocks,
                                self.cache_engine[0].block_size,
                                self.cache_engine[0].num_kv_heads // chunk_size,
                                self.cache_engine[0].head_size)

        num_blocks = len(blocks)
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((idx, blocks[idx]))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu", pin_memory=True).view(-1, 2)
        recv_cache = self.dummy_cache[:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
        # logger.info("do_recv: {} -> {},  num_blocks: {}"
        #                         .format(src_handle, self.global_rank, num_blocks))

        with self.migration_stream:
            for layer_idx in range(self.cache_engine[0].num_attention_layers):
                cache_idx = layer_idx % self.migration_num_layers
                if cache_idx == 0:
                    # logger.info("do_recv: {} -> {},  layer_idx: {}"
                    #             .format(src_handle, self.global_rank, layer_idx))
                    ss = time.time()
                    logger.info(f"self.cache_engine[0].num_attention_layers: {self.cache_engine[0].num_attention_layers},{self.migration_num_layers}")
                    logger.info(f"time[do_send] after do_recv : {time.time()-ss}")
                    if isinstance(src_worker_handle, list):
                        self.send_cache_split = list(torch.chunk(recv_cache, chunk_size, dim=3))
                        threads = []
                        logger.info(f"time[do_send] before recv_worker : {time.time()-ss}")
                        for idx, group_rank in enumerate(src_worker_handle):
                            t = threading.Thread(target=recv_worker, args=(idx, group_rank))
                            t.start()
                            threads.append(t)
                        for t in threads:
                            t.join()
                        logger.info(f"time[do_send] after recv_worker : {time.time()-ss}")
                        # 将收到的张量按照num_kv_heads进行拼接
                        cache_tmp = torch.cat(self.send_cache_split, dim=4)
                        logger.info(f"time[do_send] after cat : {time.time()-ss}")
                        cache_tmp = cache_tmp.view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
                        logger.info(f"time[do_send] after view : {time.time()-ss}")
                        recv_cache = cache_tmp
                        logger.info(f"time[do_send] after copy to  recv_cache: {time.time()-ss}")
                    else:
                        col.recv(recv_cache, src_worker_handle, self.group_name)
                    # logger.info("do_recv finished: {} -> {},  layer_idx: {}"
                    #             .format(src_handle, self.global_rank, layer_idx))
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(recv_cache[cache_idx], self.gpu_cache[virtuel_engine][layer_idx], block_mapping_tensor)
        self.migration_stream.synchronize()
        # logger.info("do_recv finished: {} -> {},  self.group_name: {}"
        #                         .format(src_handle, self.global_rank, self.group_name))


# TODO(s5u13b): Remove unnescessary args.
def get_migration_backend(instance_id: str,
                          migration_config: MigrationConfig,
                          cache_engine: List[CacheEngine],
                          worker_handle_list: List[ray.actor.ActorHandle],
                          scheduling_strategy: PlacementGroupSchedulingStrategy,
                          is_driver_worker: bool,
                          gpu_cache: Optional[List[List[torch.Tensor]]],
                          worker_rank: int,
                          local_rank: int,
                          use_ray_spmd_worker: bool,
                          worker_stage_seq_group_metadata_callback: Callable) -> MigrationBackendBase:
    assert migration_config.migration_backend in ['rayrpc', 'gloo', 'nccl'], \
        "Only support rayrpc, gloo and nccl migration backend for vLLM."

    if cache_engine[0].num_gpu_blocks < migration_config.migration_buffer_blocks:
        logger.warning("migration_cache_blocks({}) is larger than num_gpu_blocks({}), reducing it to num_gpu_blocks."
                       .format(migration_config.migration_buffer_blocks, cache_engine[0].num_gpu_blocks))
        migration_config.migration_buffer_blocks = cache_engine[0].num_gpu_blocks

    target_migration_backend = None
    backend = migration_config.migration_backend

    if backend in ['nccl', 'gloo']:
        target_migration_backend = RayColMigrationBackend(instance_id,
                                                          migration_config,
                                                          cache_engine,
                                                          local_rank,
                                                          worker_rank,
                                                          scheduling_strategy,
                                                          is_driver_worker,
                                                          gpu_cache,
                                                          use_ray_spmd_worker,
                                                          worker_stage_seq_group_metadata_callback,
                                                          )
    else:
        target_migration_backend = RayRpcMigrationBackend(instance_id,
                                                          migration_config,
                                                          cache_engine,
                                                          worker_rank,
                                                          worker_handle_list,
                                                          local_rank,
                                                          scheduling_strategy,
                                                          is_driver_worker,
                                                          gpu_cache,
                                                          use_ray_spmd_worker,
                                                          worker_stage_seq_group_metadata_callback)

    return target_migration_backend
