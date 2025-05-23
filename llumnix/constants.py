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

import torch


# examples/offline_inference.py, tests/*
MODEL_PATH: str = '/mnt/model'
DATASET_PATH: str = '/mnt/dataset'

# llumnix/manager.py
CLEAR_REQUEST_INSTANCE_INTERVAL: float = 1000.0
NO_INSTANCE_RETRY_GENERATE_INTERVAL: float = 1.0
WAIT_ALL_MIGRATIONS_DONE_INTERVAL: float = 0.1
AUTO_SCALE_UP_INTERVAL: float = 1.0
WAIT_PLACEMENT_GROUP_TIMEOUT: float = 5.0
CHECK_DEPLOYMENT_STATES_INTERVAL: float = 30.0
WATCH_DEPLOYMENT_INTERVAL: float = 10.0
INSTANCE_READY_TIMEOUT: float = 300.0
SERVER_READY_TIMEOUT: float = 60.0

# llumnix/global_scheduler/dispatch_scheduler.py
DISPATCH_LOG_FREQUENCY = 100

# llumnix/entrypoints/setup.py
MAX_RAY_RESTART_TIMES: int = 10
RAY_RESTART_INTERVAL: float = 10.0

# llumnix/entrypoints/vllm/client.py, llumnix/entrypoints/bladellm/client.py
WAIT_MANAGER_INTERVAL: float = 1.0

# llumnix/entrypoints/vllm/api_server.py
SERVER_TIMEOUT_KEEP_ALIVE: float = 5.0

# llumnix/llumlet/llumlet.py
CHECK_ENGINE_STATE_INTERVAL: float = 1.0

# llumnix/backends/vllm/llm_engine.py
NO_OUTPUTS_STEP_INTERVAL: float = 0.01

# llumnix/queue/zmq_client.py
RPC_GET_DATA_TIMEOUT_MS: int = 5000

# llumnix/queue/zmq_server.py
RPC_SOCKET_LIMIT_CUTOFF: int = 2000
RPC_ZMQ_HWM: int = 0
RETRY_BIND_ADDRESS_INTERVAL: float = 10.0
MAX_BIND_ADDRESS_RETRY_TIMES: int = 10
ZMQ_IO_THREADS: int = 8

# llumnix/entrypoints/utils.py
MAX_MANAGER_RETRY_TIMES: int = 10
RETRY_MANAGER_INTERVAL: float = 5.0
MAX_TASK_RETRIES: int = 10
RETRIES_INTERVAL: float = 5.0

# llumnix.backends/*/migration_backend.py, llumnix/backends/*/migration_worker.py
GRPC_MAX_MESSAGE_LENGTH = 1 << 31 - 1
NUMPY_SUPPORTED_DTYPES_FOR_MIGRATION = [torch.float32, torch.float16]

# llumnix/backends/vllm/llm_engine.py:_update_gpu_mertics
GPU_FIELDS_MAP = {
    # dcgmi field id
    "100": "sm_clock",
    "101": "mem_clock",

    "1001": "gr_engine_active",
    "1002": "sm_active",
    "1003": "sm_occupancy",
    "1004": "tensor_active",
    "1005": "dram_active",
    "1007": "fp32_active",
    "1008": "fp16_active",
    "1009": "pcie_tx_bytes",
    "1010": "pcie_rx_bytes",
    "1011": "nvlink_tx_bytes",
    "1012": "nvlink_rx_bytes",
    # power
    "power": "power"
}
