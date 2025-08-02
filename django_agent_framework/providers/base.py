from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional, Type, Any, Literal

from django_agent_framework.tool import ToolCall, ToolDefinition


@dataclass
class ModelMessage:
    role: Literal['system', 'user', 'assistant', 'tool']
    content: Any


@dataclass
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cached_output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class ModelResponse:
    output: Any
    usage: Usage = field(default_factory=Usage)
    tools: list[ToolCall] = field(default_factory=list)


class BaseProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        model: str,
        messages: list[ModelMessage],
        tools: Optional[list[ToolDefinition]],
        output_type: Type = str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[ModelMessage],
        tools: Optional[list[ToolDefinition]],
        output_type: Type = str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[ModelMessage]:
        """Stream response chunks from the LLM"""
        pass
