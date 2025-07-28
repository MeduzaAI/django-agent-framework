import json
import os
from typing import Optional, Type

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from openai import AsyncOpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response as OpenAIResponse,
)

from .base import BaseProvider
from django_agent_framework.llm.models import ChatMessage, Tool, LLMResponse, TokenUsage, TextResponsePart, ToolCallPart


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str = None, base_url: str = None, **kwargs):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY') or getattr(settings, 'OPENAI_API_KEY')
        self.base_url = base_url or os.environ.get('OPENAI_BASE_URL')

        if not self.api_key:
            raise ImproperlyConfigured(
                'Please set OPENAI_API_KEY environment variable if you want to use OpenAIProvider.'
            )

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)

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
        messages = [EasyInputMessageParam(content=message.content, role=message.role.value) for message in messages]
        tools = self._prepare_tools(tools)
        response = await self.client.responses.create(
            model=model, input=messages, tools=tools, max_output_tokens=max_tokens, temperature=temperature, **kwargs
        )
        return self._process_response(response)

    def _prepare_tools(self, tools: list[Tool]) -> list[FunctionToolParam]:
        return [
            FunctionToolParam(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                strict=tool.strict,
                type='function',
            )
            for tool in tools
        ]

    def _process_response(self, response: OpenAIResponse) -> LLMResponse:
        response_parts = []
        for output in response.output:
            if output.type == 'message':
                text = ''.join([content.text for content in output.content if content.type == 'output_text'])
                response_parts.append(TextResponsePart(content=text))
            if output.type == 'tool_call':
                arguments = json.loads(output.arguments)
                tool_call = ToolCallPart(tool_call_id=output.call_id, tool_name=output.name, arguments=arguments)
                response_parts.append(tool_call)

        openai_usage = response.usage.model_dump() if response.usage else {}
        usage = TokenUsage(
            input_tokens=openai_usage.get('input_tokens'),
            output_tokens=openai_usage.get('output_tokens'),
            cached_input_tokens=openai_usage.get('input_token_details', {}).get('cached_tokens'),
            cached_output_tokens=openai_usage.get('output_token_details', {}).get('cached_tokens'),
            total_tokens=openai_usage.get('total_tokens'),
        )

        return LLMResponse(
            parts=response_parts,
            usage=usage,
            model=response.model,
        )
