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

from abc import ABC, abstractmethod
from typing import List

import ray.actor

from llumnix.utils import RequestIDType


class MigrationBackendBase(ABC):
    @abstractmethod
    def init_backend(self, group_name: str, world_size: int, rank: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def destory_backend(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def recv_cache(self,
                   request_id: RequestIDType,
                   src_worker_handle: ray.actor.ActorHandle,
                   src_blocks: List[int],
                   dst_blocks: List[int],
                   is_last_stage: bool,
                      chunk_size: int=1,
                      chunk_rank: int=0) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def migrate_cache_subtract_tp(self,
                      request_id: RequestIDType,
                      src_worker_handle: List["ray.actor.ActorHandle"],
                      src_blocks: List[int],
                      dst_blocks: List[int],
                      is_last_stage: bool,
                      chunk_size: int=1,
                      chunk_rank: int=0) -> None:
        raise NotImplementedError

    @abstractmethod
    def do_send(self, request_id, dst_worker_handle: ray.actor.ActorHandle, blocks: List[int], virtuel_engine: int,  chunk_size: int=1, chunk_rank: int=0):
        raise NotImplementedError

    @abstractmethod
    def do_recv(self, request_id, src_worker_handle: ray.actor.ActorHandle, blocks: List[int], virtuel_engine: int):
        raise NotImplementedError
