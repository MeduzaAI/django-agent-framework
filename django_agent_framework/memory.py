from typing import Sequence, Union

from django_agent_framework.agent import AgentMessage
from django_agent_framework.models.agent_config import AgentConfig


class BaseMemory:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.data = []

    async def save(self, data: Union[Sequence[AgentMessage], AgentMessage]):
        self.data.append(data)

    async def read(self):
        return self.data
