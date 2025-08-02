from dataclasses import dataclass, field
from typing import Any, Union, TypedDict

from pydantic import BaseModel


class ToolDefinition(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    name: str
    tool_call_id: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool_call_id: str
    content: Any


class Tool:
    name: str
    description: str
    requires_approval: bool = False

    async def call(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        arguments = self.transform_arguments(tool_call.arguments)
        result = await self.run(**arguments, **kwargs)
        return ToolResult(tool_call_id=tool_call.tool_call_id, content=result)

    def transform_arguments(self, arguments: dict[str, Any]) -> Union[dict[str, Any], BaseModel, dataclass]:
        """Transforms argument to tools required type
        Supports:
        - dict
        - dataclasses
        - pydantic models
        """
        return arguments

    async def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def get_tool_definition(self) -> ToolDefinition:
        parameters_schema = self.get_parameters_schema()
        return {'name': self.name, 'description': self.description, 'parameters': parameters_schema}

    def get_parameters_schema(self) -> dict[str, Any]:
        """Gets the parameter schema from the tool's run method parameters."""
