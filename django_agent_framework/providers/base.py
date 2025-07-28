from abc import ABC, abstractmethod
from typing import Optional, Type, AsyncIterator
from django_agent_framework.llm.models import LLMResponse, StreamingChunk, ChatMessage, Tool


class BaseProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[Tool]],
        output_type: Type = str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[Tool]],
        output_type: Type = str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[StreamingChunk]:
        """Stream response chunks from the LLM"""
        pass
