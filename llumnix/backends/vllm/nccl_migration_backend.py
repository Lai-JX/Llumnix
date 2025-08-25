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

import os
import threading
import time
from typing import List, Tuple, Optional, Callable

from llumnix.backends.vllm.parallel_migration_backend import GroupPool, MigrationBackendResourcPool, ProxyActor
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
from llumnix.utils import RequestIDType, random_uuid, ray_get_with_timeout
import numpy as np

logger = init_logger(__name__)


def get_cur_device_id(local_rank):
    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
    return gpu_ids[local_rank]

NUMPY_SUPPORTED_DTYPES = [torch.float32, torch.float16]



class RayNCCLMigrationBackend(MigrationBackendBase):
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
        assert migration_config.migration_backend == 'nccl'
        super().__init__()
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
        self.device_id = get_cur_device_id(self.local_rank)

        self.proxy_actor = ProxyActor.options(
            scheduling_strategy=scheduling_strategy,
            name=f"ProxyActor_{self.instance_id}_{random_uuid()}").remote(
                is_driver_worker, use_ray_spmd_worker
            )
        self.gpu_cache = gpu_cache
        self.use_ray_spmd_worker = use_ray_spmd_worker
        self.worker_stage_seq_group_metadata_callback = worker_stage_seq_group_metadata_callback

        self.migration_cache_size = self.cache_engine[0].block_size * self.cache_engine[0].num_kv_heads * self.cache_engine[0].head_size

        self.cache_device = torch.device(f"cuda:{self.local_rank}")

        logger.info(f'Migration backend worker rank:{self.worker_rank}, local rank:{self.local_rank}, device id:{self.device_id}')

        self.num_migration_buffer_blocks = self.migration_config.migration_buffer_blocks
        self.num_layers = self.cache_engine[0].num_attention_layers
        self.max_migration_concurrency = self.migration_config.max_migration_concurrency

        self.buffer_pool = MigrationBackendResourcPool((self.num_migration_buffer_blocks, self.migration_num_layers, 2, self.migration_cache_size),
                                                       self.cache_engine[0].dtype,
                                                       self.cache_device,
                                                       self.local_rank,
                                                       self.max_migration_concurrency,)
        logger.info(f"buffer size {self.max_migration_concurrency} * {(self.num_migration_buffer_blocks, self.num_layers, 2, self.cache_engine[0].block_size, self.cache_engine[0].num_kv_heads, self.cache_engine[0].head_size)} * {self.cache_engine[0].dtype}")
        self.request_id_buffer = {}

    def init_backend(self, group_name: str, world_size: int, rank: int) -> bool:
        @func_set_timeout(self.migration_config.migration_backend_init_timeout)
        def init_group(world_size, rank, backend, group_name):
            col.init_collective_group(world_size, rank, backend, group_name)

        self.group_name = group_name
        self.global_world_size = world_size
        self.global_rank = rank

        self.group_pool = GroupPool(group_name, self.max_migration_concurrency)
        try:
            logger.info(
                "Initializing migration backend with group_name: {}, world_size: {}, rank: {}, local_rank: {}, device_id: {}, backend: {}".format(
                    group_name, world_size, rank, self.local_rank, self.device_id, self.backend
                )
            )

            for idx, group in self.group_pool.idx_group.items():
                init_group(world_size, rank, self.backend, group.group_name)
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
            for idx, group in self.group_pool.idx_group.items():
                col.destroy_collective_group(group.group_name)
        # pylint: disable=W0703
        except Exception as e:
            err_info = e

        if err_info is not None:
            self._log_exception("destory_backend")
        else:
            self._log_success("destory_backend")

        self.group_name = None

    # TODO
    def warmup(self) -> bool:
        if self.global_world_size > 1:
            try:
                # 用 shape=(1,) 的张量做 allreduce，保证所有进程 shape 一致
                test_tensor = torch.tensor([1.0], dtype=self.cache_engine[0].dtype, device=self.cache_device)
                for idx, group in self.group_pool.idx_group.items():
                    logger.info(f"warmup:group_name[{group.group_name}], world_size:{self.global_world_size}, rank:{self.global_rank}")
                    # col.allreduce(test_tensor, group.group_name)
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
        src_global_rank = ray_get_with_timeout(self.proxy_actor.exec_method.remote(src_worker_handle, from_driver_worker, "get_global_rank"))
        src_local_rank = ray_get_with_timeout(self.proxy_actor.exec_method.remote(src_worker_handle, from_driver_worker, "get_local_rank"))

        communication_group = self.group_pool.acquire()
        logger.info(f'recv_cache: request_id[{request_id}], communication_group[{communication_group}]')

        src_seq_group_metadata = None
        for start_idx in range(0, tot_blocks, self.num_migration_buffer_blocks):
            offset = min(self.num_migration_buffer_blocks, tot_blocks - start_idx)
            is_last_comm = (tot_blocks - start_idx <= self.num_migration_buffer_blocks)
            send_blocks = src_blocks[start_idx:start_idx+offset]
            recv_blocks = dst_blocks[start_idx:start_idx+offset]
            send_worker_metadata = self.use_ray_spmd_worker and is_last_stage and is_last_comm
            ray_obj = self.proxy_actor.exec_method.remote(
                src_worker_handle,
                from_driver_worker,
                "do_send",
                self.global_rank,
                self.worker_rank,
                send_blocks,
                request_id=request_id,
                send_worker_metadata=send_worker_metadata,
                chunk_size=chunk_size, chunk_rank=chunk_rank, communication_group=communication_group
            )
            # Ray collective communication does not have timeout parameters,
            # and run this method in another thread to set timeout will also cause cuda stream device mismatch error,
            # so recv cache does not have timeout only when the migration backend is ray collective.
            self.do_recv(request_id, src_global_rank, src_local_rank, recv_blocks, 0, 1, communication_group)
            if send_worker_metadata:
                _, src_seq_group_metadata = ray_get_with_timeout(ray_obj)
        if src_seq_group_metadata:
            self.worker_stage_seq_group_metadata_callback(request_id, src_seq_group_metadata)
        self.group_pool.release(communication_group)

    def migrate_cache_subtract_tp(self,
                      request_id: RequestIDType,
                      src_handle: List["ray.actor.ActorHandle"],
                      src_blocks: List[int],
                      dst_blocks: List[int],
                      is_last_stage: bool,
                      chunk_size: int=1) -> None:
        tot_blocks = len(src_blocks)
        from_driver_worker = (self.worker_rank // chunk_size) == 0
        tasks = []
        ss = time.time()
        for idx, handle in enumerate(src_handle):
            from_driver_worker = (idx == 0 and self.worker_rank == 0)
            tasks.append(
                self.proxy_actor.exec_method.remote(handle, from_driver_worker, "get_global_rank")
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
                    self.proxy_actor.exec_method.remote(handle, from_driver_worker, "do_send",
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

    def do_send(self, request_id, dst_global_rank, dst_local_rank, blocks: List[int], virtuel_engine: int=0, chunk_size=1, chunk_rank=0, communication_group=None):
        assert communication_group is not None
        import cupy
        num_blocks = len(blocks)
        logger.info("do_send: {} -> {}, request_id: {}, chunk_rank: {}, worker_rank:{}, local_rank:{}, num_blocks: {}"
                    .format(self.global_rank, dst_global_rank, request_id, chunk_rank, self.worker_rank, self.local_rank,num_blocks))
        if chunk_rank == 0:
            # self.barrier_actor = BarrierActor.options().remote(chunk_size)
            buffer = self.buffer_pool.acquire()
            self.request_id_buffer[request_id] = buffer
            if chunk_size > 1:
                buffer.barrier = threading.Barrier(chunk_size)
            send_cache = buffer.dummy_cache[:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
            src_to_dst: List[Tuple[int, int]] = []
            for idx in range(num_blocks):
                src_to_dst.append((blocks[idx], idx))
            block_mapping_tensor = torch.tensor(src_to_dst,
                                                dtype=torch.int64,
                                                device="cpu", pin_memory=True).view(-1, 2)

        with cupy.cuda.Device(self.local_rank):
            for layer_idx in range(self.cache_engine[0].num_attention_layers):
                cache_idx = layer_idx % self.migration_num_layers
                if chunk_rank == 0:
                    self.cache_engine[virtuel_engine].attn_backend \
                        .swap_blocks(self.gpu_cache[virtuel_engine][layer_idx], send_cache[cache_idx], block_mapping_tensor)
                    
                if cache_idx + 1 == self.migration_num_layers or layer_idx + 1 == self.cache_engine[0].num_attention_layers:
                    if chunk_size == 1:
                        with buffer.migration_stream:
                            col.send_multigpu(send_cache, dst_global_rank, dst_local_rank, communication_group.group_name)
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
                            buffer.send_cache_split = list(torch.chunk(send_cache, chunk_size, dim=4))
                            # for chunk in buffer.send_cache_split:
                            #     logger.info(f"chunk device: {chunk.device}, shape: {chunk.shape}, dtype: {chunk.dtype}")
                            # logger.info("shape after split: {} + {}; {}".format(self.send_cache_split[0].shape,self.send_cache_split[1].shape,self.migration_cache_size // chunk_size))
                            if chunk_size > 1:
                                buffer.wait_for_split_event.set()
                        else:
                            while True:
                                # 根据requst_id获取buffer
                                buffer = self.request_id_buffer.get(request_id)
                                if buffer is not None:
                                    break
                                # logger.info(f'can not find buffer for {request_id}')
                            # 等待划分完成
                            buffer.wait_for_split_event.wait()
                        
                        # logger.info("shape after split[{}]: {} + {}; {}".format(chunk_rank,self.send_cache_split[0].shape,self.send_cache_split[1].shape,self.migration_cache_size // chunk_size))
                        buffer.send_cache_split[chunk_rank] = buffer.send_cache_split[chunk_rank].reshape(
                            self.migration_num_layers, 2, num_blocks, self.migration_cache_size // chunk_size
                        )
                        
                        with buffer.migration_stream:
                            col.send_multigpu(buffer.send_cache_split[chunk_rank], dst_global_rank, dst_local_rank, communication_group.group_name)
                        buffer.barrier.wait()
                        if chunk_size > 1 and chunk_rank == 0:
                            buffer.wait_for_split_event.clear()
        if chunk_rank == 0:
            self.buffer_pool.release(buffer)
            del self.request_id_buffer[request_id]
            if chunk_size > 1:
                buffer.wait_for_split_event.clear()

    def do_recv(self, request_id, src_global_rank, src_local_rank, blocks: List[int], virtuel_engine: int=0, chunk_size=1, communication_group=None) -> None:
        assert communication_group is not None
        import cupy
        def recv_worker(idx, group_rank, worker_rank):
            with buffer.migration_stream:
                col.recv_multigpu(buffer.send_cache_split[idx], group_rank, worker_rank, communication_group.group_name)
            buffer.migration_stream.synchronize()
            buffer.send_cache_split[idx] = buffer.send_cache_split[idx].reshape(
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
        buffer = self.buffer_pool.acquire()
        self.request_id_buffer[request_id] = buffer
        recv_cache = buffer.dummy_cache[:num_blocks].view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)

        
        for layer_idx in range(self.cache_engine[0].num_attention_layers):
            cache_idx = layer_idx % self.migration_num_layers
            if cache_idx == 0:
                if isinstance(src_global_rank, list):
                    buffer.send_cache_split = list(torch.chunk(recv_cache, chunk_size, dim=3))
                    threads = []
                    for idx, group_rank, worker_rank in enumerate(zip(src_global_rank,src_local_rank)):
                        t = threading.Thread(target=recv_worker, args=(idx, group_rank, worker_rank))
                        t.start()
                        threads.append(t)
                    for t in threads:
                        t.join()
                    # 将收到的张量按照num_kv_heads进行拼接
                    cache_tmp = torch.cat(buffer.send_cache_split, dim=4)
                    # logger.info(f"time[do_send] after cat : {time.time()-ss}")
                    cache_tmp = cache_tmp.view(self.migration_num_layers, 2, num_blocks, self.migration_cache_size)
                    # logger.info(f"time[do_send] after view : {time.time()-ss}")
                    recv_cache = cache_tmp
                    # logger.info(f"time[do_send] after copy to  recv_cache: {time.time()-ss}")
                else:
                    with buffer.migration_stream:
                        col.recv_multigpu(recv_cache, src_global_rank, src_local_rank, communication_group.group_name)
            self.cache_engine[virtuel_engine].attn_backend \
                .swap_blocks(recv_cache[cache_idx], self.gpu_cache[virtuel_engine][layer_idx], block_mapping_tensor)
        del self.request_id_buffer[request_id]
        self.buffer_pool.release(buffer)