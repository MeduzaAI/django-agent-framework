from dataclasses import dataclass, field
from typing import Literal, Any, Union, Sequence

from django.core.exceptions import ImproperlyConfigured

from django_agent_framework.models import AgentConfig
from django_agent_framework.memory import BaseMemory
from django_agent_framework.providers import BaseProvider, ModelResponse, ModelMessage
from django_agent_framework.settings import agent_settings, perform_import
from django_agent_framework.tool import Tool, ToolDefinition, ToolResult, ToolCall


@dataclass
class AgentInput:
    type: Literal['text', 'image', 'file', 'audio', 'tool_result']
    content = Any


@dataclass
class ImageInput(AgentInput):
    type: Literal['image'] = 'image'


@dataclass
class FileInput(AgentInput):
    type: Literal['file'] = 'file'


@dataclass
class AudioInput(AgentInput):
    type: Literal['audio'] = 'audio'


@dataclass
class ToolResultInput(AgentInput):
    type: Literal['tool_result'] = 'tool_result'


@dataclass
class AgentOutput:
    content: Any
    tools: list[ToolCall] = field(default_factory=list)


AgentEndStrategy = Literal['early', 'exhaustive']
"""The strategy for handling multiple tool calls when a final result is found.

- `'early'`: Stop processing other tool calls once a final result is found
- `'exhaustive'`: Process all tool calls even after finding a final result
"""

AgentMessage = Union[AgentInput, ToolResult, ModelResponse]


class Agent:
    _config_model = AgentConfig
    memory_class = BaseMemory
    end_strategy: AgentEndStrategy = 'early'

    _available_tools: dict[str, Tool] = {}
    _cached_config = None

    @property
    def name(self):
        raise NotImplementedError('Agent name is required')

    @property
    async def config(self):
        if not self._cached_config:
            self._cached_config = await self._config_model.objects.aget(name=self.name)
        return self._cached_config

    @property
    async def memory(self):
        config = await self.config
        return self.memory_class(config)

    async def get_provider(self) -> BaseProvider:
        config = await self.config
        if not (provider := agent_settings.providers.get(config.provider)):
            raise ImproperlyConfigured(f'Unknown LLM provider {config.provider}')

        default_llm_provider = agent_settings.llm_providers['default']
        backend_class_import_string = provider.pop('backend') or default_llm_provider.get('backend')
        backend_class = perform_import(backend_class_import_string, f'providers.{config.provider}.backend')
        return backend_class(**provider)

    async def run(self, agent_input: list[AgentInput] = None, **kwargs) -> AgentOutput:
        return await self.model_request(agent_input, **kwargs)

    async def model_request(self, agent_input: list[AgentInput] = None, **kwargs) -> AgentOutput:
        messages = await self.prepare_messages(agent_input)
        tools = await self.get_tool_definitions()

        response = await self.generate(messages, tools, **kwargs)

        await self.save(response)

        if (response.output and self.end_strategy == 'early') or not response.tools:
            return AgentOutput(content=response.output)

        return await self.call_tools(response.tools, **kwargs)

    async def call_tools(self, tool_calls: list[ToolCall], **kwargs) -> AgentOutput:
        tools = await self.get_tools()
        tool_results = []
        tool_approval_requests: list[ToolCall] = []
        for tool_call in tool_calls:
            tool = tools.get(tool_call.name)

            if tool.requires_approval:
                tool_approval_requests.append(tool_call)
            else:
                result = await tool.call(tool_call, **kwargs)
                tool_results.append(result)

        await self.save(tool_results)

        if tool_approval_requests:
            return AgentOutput(content=None, tools=tool_approval_requests)

        return await self.model_request(tool_results, **kwargs)

    async def get_tool_definitions(self) -> list[ToolDefinition]:
        tools = await self.get_tools()
        return [tool.get_tool_definition() for tool in tools.values()]

    async def get_tools(self) -> dict[str, Tool]:
        if not self._available_tools:
            available_tools = await ToolRegistry.get_tools(self.config.tools)
            self._available_tools = available_tools
        return self._available_tools

    async def save(self, data: Union[Sequence[AgentMessage], AgentMessage]):
        memory = await self.memory
        await memory.save(data)

    async def generate(self, messages: list[ModelMessage], tools: list[ToolDefinition], **kwargs) -> ModelResponse:
        provider = await self.get_provider()
