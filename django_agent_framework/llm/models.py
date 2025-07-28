from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Literal, TypeAlias
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
class TokenUsage:
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cached_input_tokens: Optional[int]
    cached_output_tokens: Optional[int]
    total_tokens: Optional[int]


@dataclass
class PromptPart:
    content: Union[str, AudioPart, FilePart]
    role: MessageRole = MessageRole.USER


@dataclass
class TextResponsePart:
    content: str
    type: Literal['text'] = 'text'


@dataclass
class ToolCallPart:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]

    type: Literal['tool_call'] = 'tool_call'


@dataclass
class LLMResponse:
    parts: list[Union[TextResponsePart, ToolCallPart]]
    usage: Optional[TokenUsage] = None
    model: Optional[str] = None


@dataclass
class LLMRequest:
    parts: list[Union[PromptPart]]


Messages: TypeAlias = list[Union[LLMRequest, LLMResponse]]
