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
from typing import List, Tuple, Optional, Callable

from llumnix.backends.utils import BarrierActor
import torch
from func_timeout import func_set_timeout, FunctionTimedOut
import ray
import ray.util.collective as col
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm.worker.cache_engine import CacheEngine

from llumnix.internal_config import MigrationConfig
from llumnix.backends.migration_backend_interface import MigrationBackendBase
from llumnix.logging.logger import init_logger
from llumnix.constants import NUMPY_SUPPORTED_DTYPES_FOR_MIGRATION
from llumnix.utils import random_uuid
import numpy as np

logger = init_logger(__name__)

import subprocess
import threading
from queue import Queue

class MigrationBackendResource:
    """迁移后端资源"""
    def __init__(self, idx:int):

        self.idx = idx
        self.dummy_cache = torch.empty(
            size=(self.num_migration_buffer_blocks, self.num_layers, 2, self.migration_cache_size),
            dtype=self.cache_engine[0].dtype,
            device=self.cache_device,
            pin_memory=True
        )
        self.migration_stream = torch.cuda.Stream()
        self.send_cache_split = None
        self.wait_for_split_event = threading.Event()

    def use(self):
        logger.info(f"正在使用迁移后端资源: {self.idx}")

    def __str__(self):
        return f"MigrationBackendResource(idx={self.idx})"


class MigrationBackendResourcPool:
    """管理一个迁移后端资源池，方便并发迁移"""
    def __init__(self, pool_size:int = 10):
        """
        :param pool_size: 资源池大小
        """
        self.pool_size = pool_size
        self.idx_backend = dict()           #   记录每个迁移后端的索引
        self.used_backend = set()               # 记录已使用的迁移后端(idx)
        self.backend_pool = Queue()                 # 用于存储所有迁移后端的idx
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """初始化资源池，预先创建所有 MappedPortResource 并放入队列"""
        with self.lock:
            for idx in range(self):
                backend = MigrationBackendResource(idx=idx)
                self.idx_backend [idx] = backend
                self.backend_pool.put(backend.idx)  # 将迁移后端的idx放入队列

    def acquire(self):
        """
        从资源池中获取一个映射后的端口资源
        :return: svc name
        """
        with self.lock:
            if self.backend_pool.empty():
                raise RuntimeError("没有可用迁移后端了")

            backend_idx = self.backend_pool.get()
            while backend_idx in self.used_backend:  # 防止并发问题（理论上Queue已保证）
                if self.backend_pool.empty():
                    raise RuntimeError("没有可用迁移后端了")
                backend_idx = self.backend_pool.get()

            self.used_backend.add(backend_idx)
            logger.info(f"获取迁移后端: {backend_idx}")
            return backend_idx

    def release(self, backend_idx):
        """
        将映射端口释放回资源池
        :param svc_name
        """

        with self.lock:
            if backend_idx not in self.used_backend:
                raise ValueError(f"迁移后端 {backend_idx} 未被使用，无法释放")

            self.used_backend.remove(backend_idx)
            self.backend_pool.put(backend_idx)
            logger.info(f"释放迁移后端: {backend_idx}")

    def get_available_count(self):
        """获取当前可用资源数量"""
        return self.backend_pool.qsize()






@ray.remote(num_cpus=0, max_concurrency=8)
class ProxyActor:
    def __init__(self, is_driver_worker: bool, use_ray_spmd_worker: bool):
        self.is_driver_worker = is_driver_worker
        self.use_ray_spmd_worker = use_ray_spmd_worker

    def exec_method(self, handle, from_driver_worker, *args, **kwargs):

        @func_set_timeout(10)  # 10秒超时
        def _exec_method(self, handle, from_driver_worker=None, *args, **kwargs):
            if from_driver_worker is not None:
                if from_driver_worker:
                    ret = ray.get(handle.execute_engine_method_async.remote("execute_worker_method_async", *args, **kwargs))
                else:
                    ret = ray.get(handle.execute_method.options(concurrency_group="migate").remote(*args, **kwargs))
            else:
                if self.is_driver_worker and not self.use_ray_spmd_worker:
                    ret = ray.get(handle.execute_engine_method_async.remote("execute_worker_method_async", *args, **kwargs))
                else:
                    ret = ray.get(handle.execute_method.options(concurrency_group="migate").remote(*args, **kwargs))
            return ret
        
        try:
            # if from_driver_worker or (self.is_driver_worker and not self.use_ray_spmd_worker):
            #     ret = ray.get(handle.execute_engine_method_async.remote("execute_worker_method_async", *args, **kwargs))
            # else:
            #     ret = ray.get(handle.execute_method.remote(*args, **kwargs))
            ret = _exec_method(self, handle, from_driver_worker, *args, **kwargs)
        except FunctionTimedOut:
            # 打印各个参数
            raise TimeoutError(f"exec_method 超时, args: {args}, kwargs: {kwargs}")
        # pylint: disable=try-except-raise
        # except:
        #     raise

        return ret


NUMPY_SUPPORTED_DTYPES = [torch.float32, torch.float16]


class RayRpcMigrationBackend(MigrationBackendBase):
    def __init__(self,
                 instance_id: str,
                 migration_config: MigrationConfig,
                 cache_engine: List[CacheEngine],
                 worker_rank: int,
                 worker_handle_list: List["ray.actor.ActorHandle"],
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
        self.actor = ProxyActor.options(scheduling_strategy=scheduling_strategy,
                                        name=f"ProxyActor_{self.instance_id}_"+random_uuid()).remote(is_driver_worker, use_ray_spmd_worker)

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



    def init_backend(self, group_name, world_size, rank) -> bool:
        logger.info("Create rayrpc migration backend successfully.")
        return True

    def destory_backend(self) -> None:
        # rpc migration backend does not need to be destroyed as there is no group.
        # It uses ray actor handle to migration cache blocks.
        pass

    def warmup(self) -> bool:
        self.actor.exec_method.remote(self.worker_handle_list[self.worker_rank], "do_send", None, [0])
        logger.info("Rayrpc migration backend warmup successfully.")
        return True

    # The src actor will pack the kv-cache data layer by layer. Specifically, NumPy is used for the transfer
    # because, for a single node, Ray RPC can transfer NumPy arrays via shared memory. Then, the recv actor
    # first copies the data to a pinned-memory dummy cache before transferring it to the GPU to accelerate data transfer.
    def migrate_cache(self,
                      src_handle: "ray.actor.ActorHandle",
                      src_blocks: List[int],
                      dst_blocks: List[int],
                      request_id: str,
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
            ray_obj = self.actor.exec_method.remote(src_handle, from_driver_worker, "do_send",
                                                    None, send_blocks, request_id=request_id, send_worker_metadata=send_worker_metadata,
                                                    chunk_size=chunk_size, chunk_rank=chunk_rank)
            if rpc_numpy_cache is not None:
                self.do_recv(rpc_numpy_cache, recv_blocks)
            recv_blocks = dst_blocks[start_idx:start_idx+offset]

            if send_worker_metadata:
                rpc_numpy_cache, src_seq_group_metadata = ray.get(ray_obj)
            else:
                rpc_numpy_cache = ray.get(ray_obj)

        self.do_recv(rpc_numpy_cache, recv_blocks)
        if src_seq_group_metadata:
            self.worker_stage_seq_group_metadata_callback(request_id, src_seq_group_metadata)
        # ray.get(self.barrier_actor.arrive.remote())

    def migrate_cache_subtract_tp(self,
                      src_handle: List["ray.actor.ActorHandle"],
                      src_blocks: List[int],
                      dst_blocks: List[int],
                      request_id: str,
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

    def do_send(self, dst_handle: "ray.actor.ActorHandle", blocks: List[int], virtuel_engine: int=0, chunk_size=1, chunk_rank=0):
        num_blocks = len(blocks)
        ss = time.time()
        if chunk_rank == 0:
            ss = time.time()
            if chunk_size > 1:
                # self.barrier_actor = BarrierActor.options().remote(chunk_size)
                self.barrier = threading.Barrier(chunk_size)
            send_cache = self.dummy_cache[:num_blocks].view(self.num_layers, 2, num_blocks, self.migration_cache_size)
            # src_to_dst = {block_num: idx for idx, block_num in enumerate(blocks)}
            src_to_dst: List[Tuple[int, int]] = []
            for idx in range(num_blocks):
                src_to_dst.append((blocks[idx], idx))
            block_mapping_tensor = torch.tensor(src_to_dst,
                                            dtype=torch.int64,
                                            device="cpu", pin_memory=True).view(-1, 2)
            logger.info(f"time[do_send][{chunk_rank}] before swap_blocks : {time.time()-ss}")
        if chunk_rank == 0:
            with torch.cuda.stream(self.migration_stream):
                for layer_idx in range(self.num_layers):
                    self.cache_engine[virtuel_engine].attn_backend \
                        .swap_blocks(self.gpu_cache[virtuel_engine][layer_idx], send_cache[layer_idx], block_mapping_tensor)
            torch.cuda.Stream.synchronize(self.migration_stream)
            if chunk_size == 1:
                return send_cache.to(self.rpc_dtype).numpy()
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
            self.send_cache_split = list(torch.chunk(send_cache, chunk_size, dim=4))
            logger.info(f"time[do_send][{chunk_rank}] after split : {time.time()-ss}")
            # logger.info("shape after split: {} + {}; {}".format(self.send_cache_split[0].shape,self.send_cache_split[1].shape,self.migration_cache_size // chunk_size))
            if chunk_size > 1:
                self.wait_for_split_event.set()
        else:
            # 等待划分完成
            self.wait_for_split_event.wait()
        logger.info(f"time[do_send][{chunk_rank}] split finished and all process begin : {time.time()-ss}")
        
        # logger.info("shape after split[{}]: {} + {}; {}".format(chunk_rank,self.send_cache_split[0].shape,self.send_cache_split[1].shape,self.migration_cache_size // chunk_size))
        self.send_cache_split[chunk_rank] = self.send_cache_split[chunk_rank].reshape(
            self.num_layers, 2, num_blocks, self.migration_cache_size // chunk_size
        )
        logger.info(f"time[do_send][{chunk_rank}] process reshape : {time.time()-ss}")
        # ray.get(self.barrier_actor.arrive.remote())
        self.barrier.wait()
        logger.info(f"time[do_send][{chunk_rank}] after barrier_actor : {time.time()-ss}")
        if chunk_rank == 0 and chunk_size > 1:
            self.wait_for_split_event.clear()
        return self.send_cache_split[chunk_rank].to(self.rpc_dtype).numpy()

    # pylint: disable=arguments-differ
    def do_recv(self, src_handle, blocks: List[int], virtuel_engine: int=0):
        num_blocks = len(blocks)
        # src_to_dst = dict(enumerate(blocks))
        src_to_dst: List[Tuple[int, int]] = []
        for idx in range(num_blocks):
            src_to_dst.append((idx, blocks[idx]))
        block_mapping_tensor = torch.tensor(src_to_dst,
                                        dtype=torch.int64,
                                        device="cpu", pin_memory=True).view(-1, 2)
        recv_cache = self.dummy_cache[:num_blocks].view(self.num_layers, 2, num_blocks, self.migration_cache_size)
        # use pin memory dummy_cache to speed up data transfer
        recv_cache.copy_(torch.from_numpy(src_handle))

        with torch.cuda.stream(self.migration_stream):
            for layer_idx in range(self.num_layers):
                self.cache_engine[virtuel_engine].attn_backend \
                    .swap_blocks(recv_cache[layer_idx], self.gpu_cache[virtuel_engine][layer_idx], block_mapping_tensor)
        torch.cuda.Stream.synchronize(self.migration_stream)



# TODO(s5u13b): Remove unnescessary args.
def get_migration_backend(instance_id: str,
                          migration_config: MigrationConfig,
                          cache_engine: List[CacheEngine],
                          worker_handle_list: List["ray.actor.ActorHandle"],
                          scheduling_strategy: PlacementGroupSchedulingStrategy,
                          is_driver_worker: bool,
                          gpu_cache: Optional[List[List[torch.Tensor]]],
                          worker_rank: int,
                          local_rank: int,
                          use_ray_spmd_worker: bool,
                          worker_stage_seq_group_metadata_callback: Callable) -> MigrationBackendBase:
    if cache_engine[0].num_gpu_blocks < migration_config.migration_buffer_blocks:
        logger.warning("migration_cache_blocks({}) is larger than num_gpu_blocks({}), reducing it to num_gpu_blocks."
                       .format(migration_config.migration_buffer_blocks, cache_engine[0].num_gpu_blocks))
        migration_config.migration_buffer_blocks = cache_engine[0].num_gpu_blocks

    target_migration_backend = None
    backend = migration_config.migration_backend

    assert backend in ['nccl', 'rayrpc', 'gloo'], "Unsupported migration backend: {} for llumnix".format(backend)

    if backend in ['nccl', 'gloo']:
        target_migration_backend = RayColMigrationBackend(instance_id,
                                                          migration_config,
                                                          cache_engine,
                                                          local_rank,
                                                          scheduling_strategy,
                                                          is_driver_worker,
                                                          gpu_cache,
                                                          use_ray_spmd_worker,
                                                          worker_stage_seq_group_metadata_callback,
                                                          worker_rank)
    else:
        target_migration_backend = RayRpcMigrationBackend(instance_id,
                                                          migration_config,
                                                          cache_engine,
                                                          worker_rank,
                                                          worker_handle_list,
                                                          scheduling_strategy,
                                                          is_driver_worker,
                                                          gpu_cache,
                                                          use_ray_spmd_worker,
                                                          worker_stage_seq_group_metadata_callback)

    return target_migration_backend
