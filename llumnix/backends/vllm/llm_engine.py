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

from statistics import mean
import time
from typing import List, Optional, Union, Iterable, Deque, Tuple, Dict
from collections import defaultdict
import threading
import asyncio
import queue
from llumnix.constants import GPU_FIELDS_MAP
import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm.engine.async_llm_engine import _AsyncLLMEngine
from vllm.outputs import RequestOutput, RequestOutputFactory, EmbeddingRequestOutput
from vllm.sequence import SequenceGroup, SequenceStatus
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import Counter
from vllm.usage.usage_lib import UsageContext
from vllm.engine.llm_engine import SchedulerContext
from vllm import envs as vllm_envs

from llumnix.logging.logger import init_logger
from llumnix.instance_info import InstanceInfo
from llumnix.backends.backend_interface import BackendInterface, EngineState
from llumnix.backends.vllm.scheduler import SchedulerLlumnix
from llumnix.backends.vllm.sequence import SequenceGroupLlumnix, RequestStatus
from llumnix.backends.profiling import LatencyMemData
from llumnix.server_info import ServerInfo
from llumnix.internal_config import MigrationConfig
from llumnix.queue.utils import QueueType
from llumnix.backends.utils import AsyncPutQueueActor
from llumnix.utils import make_async
from llumnix.ray_utils import get_instance_name
from llumnix.llumlet.request import LlumnixRequest
from llumnix import constants
from llumnix.metrics.timestamps import set_timestamp
from llumnix.llumlet.dcgm import GPUMonitor

logger = init_logger(__name__)

NO_OUTPUTS_STEP_INTERVAL = constants.NO_OUTPUTS_STEP_INTERVAL


class LlumnixRequestOutputFactory(RequestOutputFactory):
    @staticmethod
    def create(seq_group: SequenceGroupLlumnix, use_cache: bool = False):
        # Determine the type based on a condition, for example:
        if hasattr(seq_group,
                   'embeddings') and seq_group.embeddings is not None:
            return EmbeddingRequestOutput.from_seq_group(seq_group), seq_group.server_info
        # pylint: disable=too-many-function-args
        return RequestOutput.from_seq_group(seq_group, use_cache), seq_group.server_info


class LLMEngineLlumnix(_AsyncLLMEngine):
    def __init__(self,
                 instance_id: str,
                 placement_group: PlacementGroup,
                 request_output_queue_type: QueueType,
                 disable_async_output_proc: bool,
                 *arg, **kwargs) -> None:
        # pylint: disable=import-outside-toplevel
        import vllm.outputs
        vllm.outputs.RequestOutputFactory.create = LlumnixRequestOutputFactory.create
        super().__init__(*arg, **kwargs)
        self.instance_id = instance_id
        self.step_counter = Counter()
        self.instance_info = None
        # Place the async put queue actor together with the instance.
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
        )
        self.put_queue_args_queue = queue.Queue()
        self.put_queue_loop_thread = threading.Thread(
            target=self._start_put_queue_loop, args=(), daemon=True, name="put_queue_loop"
        )
        self.async_put_queue_actor: AsyncPutQueueActor = ray.remote(
            num_cpus=1,
            scheduling_strategy=scheduling_strategy,
            name="AsyncPutQueueActor_"+instance_id
        )(AsyncPutQueueActor).remote(instance_id, request_output_queue_type)
        self.put_queue_loop_thread.start()

        self.disable_async_output_proc = disable_async_output_proc

        self._start_gpu_monitor()

    # pylint: disable=W0221
    @classmethod
    def from_engine_args(
        cls,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: EngineArgs,
        latency_mem: Optional[LatencyMemData] = None,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT
    ) -> "LLMEngineLlumnix":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()
        # Hack to pass placement_group for init workers.
        engine_config.parallel_config.placement_group = placement_group
        # Initialize the cluster and specify the executor class.
        # pylint: disable=import-outside-toplevel
        if latency_mem is not None:
            from llumnix.backends.vllm.sim_executor import SimGPUExecutor
            executor_class = SimGPUExecutor
            executor_class.latency_mem = latency_mem
        elif engine_config.parallel_config.use_ray:
            from llumnix.backends.vllm.executor import LlumnixRayGPUExecutor
            executor_class = LlumnixRayGPUExecutor
            executor_class.migration_config = migration_config
            executor_class.instance_id = instance_id
        else:
            raise ValueError('Unsupported executor backend')
        # Create the LLM engine.
        engine = cls(
            instance_id=instance_id,
            placement_group=placement_group,
            request_output_queue_type=request_output_queue_type,
            disable_async_output_proc=engine_args.disable_async_output_proc,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
        )
        return engine

    def _start_gpu_monitor(self) -> None:
        workers = self.model_executor.workers + [self.model_executor.driver_dummy_worker]
        worker_node_and_gpu_ids = [
            ray.get(worker.get_node_and_gpu_ids.remote())  # type: ignore[attr-defined]
            for worker in workers
        ]

        worker_device_ids = []
        for node_and_gpu_ids in worker_node_and_gpu_ids:
            worker_device_ids += node_and_gpu_ids[1]

        logger.info(f"worker_device_ids: {worker_device_ids};")
        self.gpuMpnitor = GPUMonitor(worker_device_ids, 1000, 10)
        self.gpuMpnitor.start()

    def _update_gpu_mertics(self, instance_info: Optional[InstanceInfo] = None) -> None:
        if self.gpuMpnitor and instance_info:
            # Get the GPU metrics from the GPU monitor.
            gpu_metrics = self.gpuMpnitor.get_gpu_metrics()
            for gpu_metric, mstric_values in gpu_metrics.items():
                mstric_value_mean_all = []
                for device_id, metric_value in mstric_values.items():
                    if len(metric_value) > 0:
                        mstric_value_mean_all.append(mean(metric_value))
                if hasattr(instance_info, GPU_FIELDS_MAP[gpu_metric]):
                    setattr(instance_info, GPU_FIELDS_MAP[gpu_metric], mstric_value_mean_all)
                    # logger.info(f"gpu metric {GPU_FIELDS_MAP[gpu_metric]}: {getattr(instance_info, GPU_FIELDS_MAP[gpu_metric])}")
                

    # pylint: disable=inconsistent-return-statements
    def _process_model_outputs(self,
                               ctx: SchedulerContext,
                               request_id: Optional[str] = None) -> None:
        if len(ctx.output_queue) == 0:
            return None

        if request_id:
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output, skip) = ctx.output_queue[0]
        else:
            (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
             is_last_step, is_first_step_output,
             skip) = ctx.output_queue.popleft()

        # Filter out outputs of migrating requests.
        server_infos = []
        if outputs:
            new_outputs = []
            new_scheduled_seq_groups = []
            new_seq_group_metadata_list = []
            for scheduled_seq_group, seq_group_meta, seq_group_output in \
                    zip(scheduler_outputs.scheduled_seq_groups, seq_group_metadata_list, outputs[0].outputs):
                seq_group = scheduled_seq_group.seq_group
                new_scheduled_seq_groups.append(scheduled_seq_group)
                new_seq_group_metadata_list.append(seq_group_meta)
                new_outputs.append(seq_group_output)
                server_infos.append(seq_group.server_info)
            scheduler_outputs.scheduled_seq_groups = new_scheduled_seq_groups
            outputs[0].outputs = new_outputs
            seq_group_metadata_list = new_seq_group_metadata_list

        if request_id:
            ctx.output_queue[0] = (outputs, seq_group_metadata_list, scheduler_outputs, is_async,
                                   is_last_step, is_first_step_output, skip)
        else:
            ctx.output_queue.appendleft((outputs, seq_group_metadata_list, scheduler_outputs, is_async,
                                         is_last_step, is_first_step_output, skip))

        set_timestamp(server_infos, 'engine_process_model_outputs_timestamp_begin', time.time())

        super()._process_model_outputs(ctx, request_id)

        if ctx.request_outputs:
            request_outputs, server_infos = zip(*ctx.request_outputs)

            for request_output, server_info in zip(request_outputs, server_infos):
                if hasattr(server_info, 'request_timestamps'):
                    request_output.request_timestamps = server_info.request_timestamps
            set_timestamp(request_outputs, 'engine_process_model_outputs_timestamp_end', time.time())

        if not self.disable_async_output_proc:
            while self._output_proc_done_event_queue.qsize() > 0:
                output_proc_done_event = self._output_proc_done_event_queue.get()
                output_proc_done_event.set()

        return

    def _process_request_outputs(
            self,
            outputs: List[Tuple[RequestOutput, ServerInfo]],
    ) -> Tuple[List[RequestOutput], List[ServerInfo]]:
        request_outputs = []
        server_infos = []
        if outputs:
            request_outputs, server_infos = zip(*outputs)
            request_outputs = list(request_outputs)
            server_infos = list(server_infos)

        set_timestamp(request_outputs, 'engine_step_timestamp_begin', self.step_begin_time)
        set_timestamp(request_outputs, 'engine_step_timestamp_end', time.time())
        for request_output, server_info in zip(request_outputs, server_infos):
            if server_info.request_timestamps.migrate_out_one_request_end == 0.0:
                logger.info("[LJX] LLMEngineLlumnix._process_request_outputs engine_step_timestamp_end, {}, timestamps: {}".format(request_output.request_id, time.time()))

        for request_output in request_outputs:
            if request_output.finished:
                logger.info("engine finished request {}".format(request_output.request_id))

        instance_info: InstanceInfo = self.instance_info
        instance_info.instance_id = self.instance_id
        instance_info.step_id = next(self.step_counter)
        instance_info.timestamp = time.time()
        # TODO(KuilongCui): add cli_args to determine whether to collect profiling data
        instance_info.profiling_data=(instance_info.inference_type.value if instance_info.inference_type else "",
                                      instance_info.num_seqs,
                                      sum(instance_info.running_seq_lens),
                                      self.model_executor.last_inference_latency)
        seq_groups = self.scheduler[0].running
        if seq_groups:
            tot_blocks = []
            for seq in seq_groups[-1].get_seqs(SequenceStatus.RUNNING):
                blocks = self.scheduler[0].block_manager.get_block_table(seq)
                tot_blocks.extend(blocks)
            tot_blocks = set(tot_blocks)
            instance_info.num_blocks_last_running_request = len(tot_blocks)

        self.instance_info = instance_info

        set_timestamp(request_outputs, 'engine_put_queue_timestamp', time.time())

        if request_outputs:
            self.put_queue_args_queue.put_nowait((request_outputs, server_infos))

        set_timestamp(request_outputs, 'engine_step_postprocess_timestamp_end', time.time())

        return request_outputs, server_infos

    async def step_async(self) -> Tuple[List[RequestOutput], List[ServerInfo]]:
        self.step_begin_time = time.time()
        # pylint: disable=too-many-function-args
        outputs = await super().step_async(0)
        return self._process_request_outputs(outputs)

    def update_instance_info(self, instance_info: InstanceInfo) -> None:
        # These fields are updated after step.
        if self.instance_info is not None:
            instance_info.instance_id = self.instance_info.instance_id
            instance_info.step_id = self.instance_info.step_id
            instance_info.timestamp = self.instance_info.timestamp
            instance_info.profiling_data = self.instance_info.profiling_data
            instance_info.num_blocks_last_running_request = self.instance_info.num_blocks_last_running_request
            self._update_gpu_mertics(instance_info)
        self.instance_info = instance_info

    # pylint: disable=invalid-overridden-method
    async def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, *args, **kwargs):
        super().add_request(request_id, *args, **kwargs)
        seq_group = self.scheduler[0].waiting[-1]
        set_timestamp(server_info, 'engine_add_request_timestamp', time.time())
        self.scheduler[0].waiting[-1] = SequenceGroupLlumnix(request_id, server_info, expected_steps, [seq_group.get_seqs()[0]],
                                                             seq_group.metrics.arrival_time, seq_group.sampling_params, seq_group.lora_request,
                                                             seq_group.trace_headers, seq_group.prompt_adapter_request, seq_group.encoder_seq,
                                                             seq_group.priority)

    def _start_put_queue_loop(self):
        while True:
            args = self.put_queue_args_queue.get()
            request_outputs, server_infos = args
            set_timestamp(request_outputs, 'engine_thread_put_queue_timestamp', time.time())
            self._put_request_outputs_to_server(request_outputs, server_infos)

    def _put_request_outputs_to_server(self, request_outputs: List[RequestOutput], server_infos: List[ServerInfo]) -> None:
        server_request_outputs = defaultdict(list)
        server_info_dict = {}
        # Reorganize data in orther to put request output to queue in batch at one time.
        for request_output, server_info in zip(request_outputs, server_infos):
            server_id = server_info.server_id
            server_request_outputs[server_id].append(request_output)
            if server_id not in server_info_dict:
                server_info_dict[server_id] = server_info
        # TODO(s5u13b): Reduce the across-actor overhead.
        if server_info_dict:
            self.async_put_queue_actor.put_nowait_to_servers.remote(server_request_outputs, server_info_dict)


class BackendVLLM(BackendInterface):
    def __init__(
        self,
        instance_id: str,
        placement_group: PlacementGroup,
        request_output_queue_type: QueueType,
        migration_config: MigrationConfig,
        engine_args: EngineArgs,
    ) -> None:
        self.engine: LLMEngineLlumnix = LLMEngineLlumnix.from_engine_args(engine_args=engine_args,
                                                                          request_output_queue_type=request_output_queue_type,
                                                                          migration_config=migration_config,
                                                                          instance_id=instance_id,
                                                                          placement_group=placement_group)
        # In order to call the verify_async_output_proc implicitly.
        engine_config = engine_args.create_engine_config()
        if not engine_config.model_config.use_async_output_proc:
            self.engine.scheduler = [SchedulerLlumnix(self.engine.scheduler_config, self.engine.cache_config, self.engine.lora_config)
                                     for _ in range(engine_args.pipeline_parallel_size)]
        else:
            self.engine.scheduler = [
                SchedulerLlumnix(self.engine.scheduler_config, self.engine.cache_config, self.engine.lora_config,
                                engine_args.pipeline_parallel_size, self.engine.async_callbacks[v_id])
                                for v_id in range(engine_args.pipeline_parallel_size)]
        for vid in range(engine_args.pipeline_parallel_size):
            self.engine.scheduler[vid].add_update_instance_info_callback(self.engine.update_instance_info)
        self.engine.output_processor.scheduler = self.engine.scheduler
        self.migration_config = migration_config
        self.instance_id = instance_id
        self.worker_handle_list = self.engine.model_executor.workers.copy()
        if len(self.worker_handle_list) + 1 == self.engine.parallel_config.world_size:
            self.worker_handle_list.insert(0, ray.get_actor(get_instance_name(self.instance_id), namespace="llumnix"))
        self._run_workers("init_migration", instance_id=instance_id,
                                            migration_config=migration_config,
                                            src_worker_handle_list=self.worker_handle_list,
                                            placement_group=placement_group)

        self.state = EngineState.INIT
        logger.info("engine ({}) current state {}".format(self.instance_id, self.state))

        self.disable_async_output_proc = engine_args.disable_async_output_proc

        self._step_done_event_queue = queue.Queue()
        self._remove_running_request_ret: Dict[str] = {}
        if not self.disable_async_output_proc:
            self._output_proc_done_event_queue = queue.Queue()
            self.engine._output_proc_done_event_queue = self._output_proc_done_event_queue

        self.use_ray_spmd_worker = vllm_envs.VLLM_USE_RAY_SPMD_WORKER

        self._stop_event = asyncio.Event()
        asyncio.create_task(self._start_engine_step_loop())

    async def _start_engine_step_loop(self) -> None:
        self._stop_event.clear()

        previous_state = self.state
        self.state = EngineState.RUNNING
        logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))

        while not self._stop_event.is_set():
            try:
                while self._step_done_event_queue.qsize() > 0:
                    request_id, step_done_event = self._step_done_event_queue.get()
                    self._remove_running_request_ret[request_id] = self._remove_running_request(request_id)
                    step_done_event.set()
                await asyncio.sleep(0.0)
                request_outputs, _ = await self.engine.step_async()
                # logger.info("[LJX] step_async: {}, seq_id".format(time.time()))
                if len(request_outputs) == 0:
                    await asyncio.sleep(NO_OUTPUTS_STEP_INTERVAL)
            # pylint: disable=broad-except
            except Exception as e:
                logger.exception("Error in engine loop: {}".format(e))
                self._run_workers("shutdown")

                previous_state = self.state
                self.state = EngineState.CRASHED
                logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, previous_state, self.state))
                break

        if self.state == EngineState.RUNNING:
            self.state = EngineState.STOPPED
            logger.info("engine ({}) change state: {} -> {}".format(self.instance_id, EngineState.RUNNING, self.state))

    async def execute_worker_method_async(self, method, *args, **kwargs):
        return await make_async(self.engine.model_executor.driver_worker.execute_method)(method, *args, **kwargs)

    # Store the server information of each request to put the request outputs back to the corresponding api server correctly.
    async def add_request(self, request_id: str, server_info: ServerInfo, expected_steps: int, *args, **kwargs) -> None:
        await self.engine.add_request(request_id, server_info, expected_steps, *args, **kwargs)

    async def commit_dst_request(self, backend_request: SequenceGroupLlumnix) -> None:
        if self.use_ray_spmd_worker and backend_request.status == RequestStatus.RUNNING_MIGRATING:
            await self._run_workers_async("commit_seq_group_metadata", backend_request.request_id)

        seq = backend_request.get_seqs()[0]
        seq.seq_id = next(self.engine.seq_counter)
        logger.info("pop request {} from pre_alloc_cache_dict,{}".format(backend_request.request_id,time.time()))
        pre_alloc_blocks = self.engine.scheduler[0].pre_alloc_cache_dict.pop(backend_request.request_id)
        self.engine.scheduler[0].block_manager.add_block_table(pre_alloc_blocks, seq.seq_id)
        backend_request.reset_migration_args_dst()
        logger.info("[LJX] request ({}) reset_migration_args_dst: {}".format(backend_request.request_id, time.time()))
        assert RequestStatus.is_migrating(backend_request.status), \
            "The status of request migrated to dst instance should be  \
             RequestStatus.WAITING_MIGRATING or RequestStatus.RUNNING_MIGRATING"
        if backend_request.status == RequestStatus.RUNNING_MIGRATING:
            backend_request.reset_status()
            set_timestamp(backend_request, 'migrated_add_running_request_start', time.time())
            self.add_running_request(backend_request)
            set_timestamp(backend_request, 'migrated_add_running_request_end', time.time())
        else: # WAITING_MIGRATING:
            self.add_waiting_request(backend_request)

    async def send_blocks(self,
                          dst_ray_actor: "ray.actor.ActorHandle",
                          src_blocks: List[int],
                          dst_blocks: List[int],
                          request_id: str,
                          is_last_stage: bool) -> None:     # dst_ray_actor: Llumlet ActorHandle
        await dst_ray_actor.execute_engine_method_async.remote("_run_workers_async",
                                                               "migrate_cache",
                                                               src_worker_handle_list=self.worker_handle_list,
                                                               dst_blocks=dst_blocks,
                                                               src_blocks=src_blocks,
                                                               request_id=request_id,
                                                               is_last_stage=is_last_stage)

    def _run_workers(self, *args, **kwargs):
        # pylint: disable=protected-access
        return self.engine.model_executor._run_workers(*args, **kwargs)

    async def _run_workers_async(self, *args, **kwargs):
        # pylint: disable=protected-access
        return await make_async(self.engine.model_executor._run_workers)(*args, **kwargs)

    async def is_ready(self):
        return True

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        return self.engine.abort_request(request_ids)

    def get_running_queue(self) -> List[SequenceGroupLlumnix]:
        return self.engine.scheduler[0].get_running_queue()

    def get_waiting_queue(self) -> Deque[SequenceGroupLlumnix]:
        return self.engine.scheduler[0].get_waiting_queue()

    async def get_request_incremental_blocks(self, backend_request: LlumnixRequest, pre_stage_num_blocks: int) -> Tuple[List[int], List[int]]:
        incremental_blocks, incremental_token_ids = \
            self.engine.scheduler[0].get_request_incremental_blocks(backend_request, pre_stage_num_blocks)
        is_last_stage = (len(incremental_blocks) <= self.migration_config.migration_last_stage_max_blocks) or backend_request.blocking_migration
        return incremental_blocks, incremental_token_ids, is_last_stage

    # 迁移时需要调用到
    async def remove_running_request(self, request_id: str) -> bool:
        step_done_event = asyncio.Event()
        self._step_done_event_queue.put((request_id, step_done_event))
        await step_done_event.wait()
        ret = self._remove_running_request_ret.pop(request_id)
        if not self.disable_async_output_proc:
            output_proc_done_event = asyncio.Event()
            self._output_proc_done_event_queue.put(output_proc_done_event)
            await output_proc_done_event.wait()
        return ret

    def _remove_running_request(self, request_id: str) -> bool:
        return self.engine.scheduler[0].remove_running_request(request_id)

    def remove_waiting_request(self, *args, **kwargs) -> bool:
        return self.engine.scheduler[0].remove_waiting_request(*args, **kwargs)

    def add_migrating_out_request_last_stage(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].add_migrating_out_request_last_stage(*args, **kwargs)

    def pop_migrating_out_request_last_stage(self, backend_request: LlumnixRequest) -> None:
        # Only running requests have sequence group metadata in workers.
        if self.use_ray_spmd_worker and backend_request.status == RequestStatus.RUNNING_MIGRATING:
            # pylint: disable=protected-access
            asyncio.create_task(
                self._run_workers_async(
                    "pop_migrating_out_seq_group_metadata", backend_request.request_id))
        return self.engine.scheduler[0].pop_migrating_out_request_last_stage(backend_request.request_id)

    def free_migrating_out_requests_last_stage(self, *args, **kwargs) -> List[LlumnixRequest]:
        migrating_out_requests_last_stage = self.engine.scheduler[0].free_migrating_out_requests_last_stage(*args, **kwargs)
        if self.use_ray_spmd_worker and migrating_out_requests_last_stage and \
            migrating_out_requests_last_stage[0].status == RequestStatus.RUNNING_MIGRATING:
            # pylint: disable=protected-access
            asyncio.create_task(self._run_workers_async("restore_migrating_out_seq_group_metadata"))
        return migrating_out_requests_last_stage

    def pre_alloc(self, *args, **kwargs) -> List[int]:
        return self.engine.scheduler[0].pre_alloc(*args, **kwargs)

    def should_abort_migration(self, *args, **kwargs) -> bool:
        return self.engine.scheduler[0].should_abort_migration(*args, **kwargs)

    def add_running_request(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].add_running_request(*args, **kwargs)

    def add_waiting_request(self, *args, **kwargs) -> None:
        return self.engine.scheduler[0].add_waiting_request(*args, **kwargs)

    def is_request_running(self, *args, **kwargs) -> bool:
        return self.engine.scheduler[0].is_request_running(*args, **kwargs)

    def free_dst_pre_alloc_cache(self, request_id: str = None) -> None:
        # request is None when free_dst_pre_alloc_cache is called by clear_migration_states.
        # TODO(s5u13b): Only needed when running waiting request.
        if request_id is None and self.use_ray_spmd_worker:
            # pylint: disable=protected-access
            asyncio.create_task(self._run_workers_async("free_migrating_in_seq_group_metadata"))
        return self.engine.scheduler[0].free_dst_pre_alloc_cache(request_id)

    def free_src_request(self, backend_request: SequenceGroup) -> None:
        return self.engine.scheduler[0].free_src_request(backend_request)

    def get_all_request_ids(self) -> List[str]:
        return self.engine.scheduler[0].get_all_request_ids()
