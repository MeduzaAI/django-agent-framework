from abc import ABC, abstractmethod
from typing import Iterator, Optional, Type
from ..models import LLMResponse, StreamingChunk, ChatMessage, Tool


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(
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
    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: Optional[list[Tool]],
        output_type: Type = str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[StreamingChunk]:
        """Stream response chunks from the LLM"""
        pass
