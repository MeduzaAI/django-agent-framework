from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from enum import Enum
import json

T = TypeVar('T')


class MessageRole(Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    TOOL = 'tool'


class ResponseFormat(Enum):
    TEXT = 'text'
    JSON_OBJECT = 'json_object'
    JSON_SCHEMA = 'json_schema'


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class ChatMessage:
    role: MessageRole
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    strict: bool = False


@dataclass
class StructuredOutputSchema:
    name: str
    description: Optional[str]
    schema: Dict[str, Any]
    strict: bool = True


@dataclass
class TokenUsage:
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cached_input_tokens: Optional[int]
    cached_output_tokens: Optional[int]
    total_tokens: Optional[int]


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[TokenUsage] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    structured_output: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None

    def parse_structured_output(self, target_class: Type) -> Optional[Any]:
        '''Parse structured output into a specific dataclass or type'''
        if not self.structured_output:
            return None

        if hasattr(target_class, '__dataclass_fields__'):
            return target_class(**self.structured_output)

        return target_class(self.structured_output)


@dataclass
class StreamingChunk:
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None
    structured_output: Optional[Dict[str, Any]] = None


@dataclass
class LLMRequest:
    messages: List[ChatMessage]
    model: str
    tools: Optional[List[Tool]] = None
    response_format: ResponseFormat = ResponseFormat.TEXT
    structured_output_schema: Optional[StructuredOutputSchema] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
