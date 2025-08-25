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
from typing import List, Union, Iterable
import time

import ray
import ray.actor
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo, InstanceLoadCalculator, InstanceType
from llumnix.backends.backend_interface import BackendInterface
from llumnix.backends.utils import init_backend_engine, EngineState
from llumnix.llumlet.migration_coordinator import MigrationCoordinator
from llumnix.server_info import ServerInfo
from llumnix.queue.queue_type import QueueType
from llumnix.arg_utils import InstanceArgs, LlumnixEngineArgs
from llumnix.ray_utils import get_instance_name, log_actor_ray_info
from llumnix.constants import CHECK_ENGINE_STATE_INTERVAL
from llumnix.metrics.timestamps import set_timestamp
from llumnix.metrics.llumlet_metrics import LlumletMetrics
from llumnix.utils import RequestIDType, BackendType
from llumnix.constants import NUM_GPUS_VLLM_GPU_ACTOR, NUM_GPUS_BLADELLM_GPU_ACTOR

logger = init_logger(__name__)


class Llumlet:
    def __init__(self,
                 instance_id: str,
                 instance_args: InstanceArgs,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 llumnix_engine_args: LlumnixEngineArgs) -> None:
        log_actor_ray_info(actor_class_name=self.__class__.__name__)
        self.instance_id = instance_id
        logger.info("Llumlet(instance_id={}, backend_type={}, instance_type={})".format(
            self.instance_id, llumnix_engine_args.backend_type, instance_args.instance_type))
        self.instance_args: InstanceArgs = instance_args
        self.enable_migration = instance_args.enable_migration
        self.actor_name = get_instance_name(instance_id)

        self.instance_load_calculator = InstanceLoadCalculator(instance_args=self.instance_args)

        self.backend_engine: BackendInterface = init_backend_engine(self.instance_id,
                                                                    placement_group,
                                                                    request_output_queue_type,
                                                                    instance_args,
                                                                    llumnix_engine_args)
        if self.enable_migration:
            self.migration_coordinator = MigrationCoordinator(
                self.instance_id,
                self.backend_engine,
                llumnix_engine_args.backend_type,
                instance_args.max_migration_concurrency,
                instance_args.request_migration_policy,
                instance_args.migration_last_stage_max_blocks,
                instance_args.migration_max_stages,
            )
        self.llumlet_metrics = LlumletMetrics()

        asyncio.create_task(self._check_engine_state_loop())

    def __repr__(self):
        return f"{self.__class__.__name__}(iid={self.instance_id[:5]})"

    @classmethod
    def from_args(cls,
                  instance_id: str,
                  instance_args: InstanceArgs,
                  placement_group: PlacementGroup,
                  request_output_queue_type: QueueType,
                  engine_args: LlumnixEngineArgs):
        backend_type = engine_args.backend_type
        assert backend_type in [BackendType.VLLM, BackendType.BLADELLM, BackendType.VLLM_V1, BackendType.SIM_VLLM], \
            f'unimplemented backend {BackendType}'
        # There could be some cuda related imports or codes inside the llm engine of llumlet, so we allocate gpu to llumlet.
        if backend_type == BackendType.VLLM:
            num_gpus = NUM_GPUS_VLLM_GPU_ACTOR
        elif backend_type == BackendType.BLADELLM:
            num_gpus = NUM_GPUS_BLADELLM_GPU_ACTOR
        else: # backend_type == BackendType.SIM_VLLM
            num_gpus = 0
        llumlet_class = ray.remote(num_cpus=1,
                                    num_gpus=num_gpus,
                                    name=get_instance_name(instance_id),
                                    namespace='llumnix',
                                    lifetime="detached")(cls).options(
                                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                                            placement_group=placement_group,
                                            placement_group_bundle_index=0,
                                            placement_group_capture_child_tasks=True
                                        )
                                    )
        llumlet = llumlet_class.remote(instance_id,
                                       instance_args,
                                       placement_group,
                                       request_output_queue_type,
                                       engine_args)

        return llumlet

    async def _check_engine_state_loop(self):
        while True:
            await asyncio.sleep(CHECK_ENGINE_STATE_INTERVAL)
            if self.backend_engine.state == EngineState.CRASHED:
                logger.error("Llumlet {} detected backend engine crashed. Stopping...".format(self.instance_id))
                # pylint: disable=protected-access
                self.backend_engine._stop_event.set()
                await asyncio.sleep(0)
                self.stop()

    def stop(self):
        if self.backend_engine.state == EngineState.RUNNING:
            self.backend_engine.stop()
        self_actor = ray.get_actor(name=self.actor_name, namespace="llumnix")
        ray.kill(self_actor)

    # async def migrate_out(self, dst_instance_id: str, dst_instance_actor_handle: ray.actor.ActorHandle) -> List[str]:
    #     migrate_out_requests = self.migration_scheduler.get_migrate_out_requests()

    #     if len(migrate_out_requests) == 0:
    #         return []

    #     for migrate_out_request in migrate_out_requests:
    #         migrate_out_request.is_migrating = True

    #     migrated_request_list = []
    #     logger.info("[LJX] Llumlet._migrate_out start, timestamps: {}".format(time.time()))
    #     for migrate_out_request in migrate_out_requests:

    #         migrate_out_one_request_begin = time.time()
    #         logger.info("[LJX] Llumlet._migrate_out_one_request start, {}, timestamps: {}".format(migrate_out_request.request_id, migrate_out_one_request_begin))
    #         set_timestamp(migrate_out_request.server_info, "migrate_out_one_request_begin", time.time())
            
    #         migrated_request = await self._migrate_out_one_request(migrate_out_request, dst_instance_id, dst_instance_actor_handle)
            
    #         migrate_out_one_request_end = time.time()
    #         logger.info("[LJX] Llumlet._migrate_out_one_request end, {}, timestamps: {}".format(migrate_out_request.request_id, migrate_out_one_request_end))
    #         logger.info("[LJX] Llumlet._migrate_out_one_request latency: {} ms".format((migrate_out_one_request_end - migrate_out_one_request_begin)*1000))
            
    #         migrated_request_list.extend(migrated_request)
    #         if len(migrated_request) == 0 and migrate_out_request.eom:
    #             break
    #     logger.info("[LJX] Llumlet._migrate_out end, timestamps: {}".format(time.time()))

    #     return migrated_request_list
    

    # async def migrate_out(self, dst_instance_id: str, dst_instance_actor_handle: ray.actor.ActorHandle) -> List[str]:
    #     migrate_out_requests = self.migration_scheduler.get_migrate_out_requests()

    #     if len(migrate_out_requests) == 0:
    #         return []

    #     for migrate_out_request in migrate_out_requests:
    #         migrate_out_request.is_migrating = True

    #     migrated_request_list = []
    #     logger.info("[LJX] Llumlet._migrate_out start, timestamps: {}".format(time.time()))

    #     tasks = []
    #     for migrate_out_request in migrate_out_requests:
    #         migrate_out_request.is_migrating = True
    #         migrate_out_one_request_begin = time.time()
    #         logger.info("[LJX] Llumlet._migrate_out_one_request start, {}, timestamps: {}".format(migrate_out_request.request_id, migrate_out_one_request_begin))
    #         set_timestamp(migrate_out_request.server_info, "migrate_out_one_request_begin", time.time())
    #         tasks.append(self._migrate_out_one_request(migrate_out_request, dst_instance_id, dst_instance_actor_handle))

    #     # 并发执行所有迁移
    #     results = await asyncio.gather(*tasks)

    #     for migrated_request, migrate_out_request in zip(results, migrate_out_requests):
    #         migrate_out_one_request_end = time.time()
    #         logger.info("[LJX] Llumlet._migrate_out_one_request end, {}, timestamps: {}".format(migrate_out_request.request_id, migrate_out_one_request_end))
    #         logger.info("[LJX] Llumlet._migrate_out_one_request latency: {} ms".format((migrate_out_one_request_end - migrate_out_one_request_begin)*1000))
    #         migrated_request_list.extend(migrated_request)
    #         if len(migrated_request) == 0 and migrate_out_request.eom:
    #             break

    #     logger.info("[LJX] Llumlet._migrate_out end, timestamps: {}".format(time.time()))
    #     return migrated_request_list

    # async def _migrate_out_one_request(self,
    #                                    migrate_out_request: LlumnixRequest,
    #                                    dst_instance_id: str,
    #                                    dst_instance_actor_handle: ray.actor.ActorHandle) -> List[LlumnixRequest]:
    #     try:
    #         t0 = time.time()
    #         logger.info("{}->{} begin migrate out".format(self.instance_id, dst_instance_id))
    #         migrated_request = []

    #         if migrate_out_request.status == RequestStatus.RUNNING:
    #             migrate_out_request.migration_start_time = time.time()
    #             status = await self.migration_coordinator.migrate_out_running_request(dst_instance_actor_handle, migrate_out_request)
    #         elif migrate_out_request.status == RequestStatus.WAITING:
    #             migrate_out_request.migration_start_time = time.time()
    #             status = await self.migration_coordinator.migrate_out_waiting_request(dst_instance_actor_handle, migrate_out_request)
    #         else:
    #             return migrated_request

    #         if status == MigrationStatus.FINISHED:
    #             set_timestamp(migrate_out_request.server_info, "migrate_out_one_request_end", time.time())
    #             await dst_instance_actor_handle.execute_engine_method_async.remote("commit_dst_request", migrate_out_request)
    #             self.backend_engine.free_src_request(migrate_out_request)
    #             self.backend_engine.pop_migrating_out_request_last_stage(migrate_out_request)
    #             migrated_request.append(migrate_out_request.request_id)
    #         else: # ABORTED_SRC or ABORTED_DST
    #             migrate_out_request.reset_migration_args_src()
    #             migrate_out_request.reset_status()
    #             # If dst aborts itself, dst proactively frees the pre allocated cache in migrate_in_pre_alloc.
    #             if status == MigrationStatus.ABORTED_SRC:
    #                 await dst_instance_actor_handle.execute_migration_method.remote("free_dst_pre_alloc_cache", migrate_out_request.request_id)
    #         t1 = time.time()
    #         logger.info("Instance {}->{} migrate done, migrate request {}, migration status: {}, len: {} blocks, cost: {} ms" \
    #                     .format(self.instance_id, dst_instance_id, migrated_request, status, \
    #                             sum(migrate_out_request.stage_num_blocks_list), (t1 - t0)*1000))
    #     except ray.exceptions.RayActorError:
    #         logger.info("Instance {} is dead.".format(dst_instance_id))
    #         raise
    #     # pylint: disable=W0703
    #     except Exception as e:
    #         logger.exception("Unexpected exception: {}".format(e))
    #         raise
    #     return migrated_request

    # TODO(KuilongCui): only the metrics-related information needs to be synchronously loaded for the manager
    def get_instance_info(self) -> InstanceInfo:
        instance_info: InstanceInfo = self.backend_engine.get_instance_info()
        instance_info.instance_type = self.instance_args.instance_type
        instance_info.enable_defrag = self.instance_args.enable_defrag
        self.instance_load_calculator.compute_instance_load(instance_info)
        if self.enable_migration:
            instance_info.has_migration_slot = self.migration_coordinator.has_migration_slot()
            instance_info.is_migrating = self.migration_coordinator.is_migrating()
        return instance_info

    async def is_ready(self):
        await self.backend_engine.is_ready()
        return True

    async def get_instance_type(self) -> InstanceType:
        await self.backend_engine.is_ready()
        return self.instance_args.instance_type

    async def get_engine_disagg_inst_id(self) -> str:
        await self.backend_engine.is_ready()
        return self.backend_engine.engine_disagg_inst_id

    async def generate(self, request_id: RequestIDType, server_info: ServerInfo, expected_steps: int, *args, **kwargs) -> None:
        set_timestamp(server_info, 'llumlet_generate_timestamp', time.time())
        self.llumlet_metrics.llumlet_request_qps.increase(
            labels={"instance_id": self.instance_id}
        )
        await self.backend_engine.add_request(
            request_id, server_info, expected_steps, *args, **kwargs
        )

    async def abort(self, request_id: Union[RequestIDType, Iterable[RequestIDType]]) -> None:
        if isinstance(request_id, (str, int)):
            request_id = (request_id,)
        request_ids = set(request_id)
        await self.backend_engine.abort_request(request_ids)

    async def migrate_out(self, dst_instance_actor: ray.actor.ActorHandle, dst_instance_id: str) -> List[RequestIDType]:
        return await self.migration_coordinator.migrate_out(dst_instance_actor, dst_instance_id)

    def execute_engine_method(self, method, *args, **kwargs):
        executor = getattr(self.backend_engine, method)
        return executor(*args, **kwargs)

    async def execute_engine_method_async(self, method, *args, **kwargs):
        executor = getattr(self.backend_engine, method)
        return await executor(*args, **kwargs)
    
    def get_world_size(self):
        return self.backend_engine.engine.parallel_config.world_size

    def execute_migration_method(self, method, *args, **kwargs):
        executor = getattr(self.migration_coordinator, method)
        return executor(*args, **kwargs)

    async def execute_migration_method_async(self, method, *args, **kwargs):
        executor = getattr(self.migration_coordinator, method)
        return await executor(*args, **kwargs)
    def get_class_name(self):
        return self.__class__.__name__
