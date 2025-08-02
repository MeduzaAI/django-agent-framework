import os
from typing import Optional, Type

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from openai.types.responses import (
    ResponseInputItemParam,
    EasyInputMessageParam,
    FunctionToolParam,
    Response as OpenAIResponse,
)

from .base import BaseProvider
from openai import OpenAI


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str = None, base_url: str = None, **kwargs):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY') or getattr(settings, 'OPENAI_API_KEY')
        self.base_url = base_url or os.environ.get('OPENAI_BASE_URL')

        if not self.api_key:
            raise ImproperlyConfigured(
                'Please set OPENAI_API_KEY environment variable if you want to use OpenAIProvider.'
            )

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

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
        messages = self._prepare_messages(messages)
        tools = self._prepare_tools(tools)
        response = self.client.responses.create(
            model=model, input=messages, tools=tools, max_output_tokens=max_tokens, temperature=temperature, **kwargs
        )
        return self._process_response(response)

    def _prepare_messages(self, messages: list[ChatMessage]) -> list[ResponseInputItemParam]:
        return [EasyInputMessageParam(content=message.content, role=message.role.value) for message in messages]

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
                text_message = self._parse_text_message(output)
                response_parts.append(text_message)
            if output.type == 'tool_call':
                tool_call = self._parse_tool_call(output)
                response_parts.append(tool_call)

        openai_usage = response.usage.model_dump() if response.usage else {}
        usage = TokenUsage(
            input_tokens=openai_usage.get('input_tokens'),
            output_tokens=openai_usage.get('output_tokens'),
            cached_input_tokens=openai_usage.get('input_token_details', {}).get('cached_tokens'),
            cached_output_tokens=openai_usage.get('output_token_details', {}).get('cached_tokens'),
            total_tokens=openai_usage.get('total_tokens'),
        )
