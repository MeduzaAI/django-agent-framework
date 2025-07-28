from django.core.exceptions import ImproperlyConfigured

from django_agent_framework.llm.models import ChatMessage
from django_agent_framework.providers.base import BaseProvider
from django_agent_framework.settings import agent_settings, perform_import


class Memory:
    def get(self, *args, **kwargs): ...

    def update(self, *args, **kwargs): ...


class Agent:
    message_model = Message
    message_thread_model = MessageThread
    memory_class = Memory

    def __init__(self, message_thread_id):
        self.message_thread_id = message_thread_id

    @property
    def config(self):
        return {'provider': 'openai', 'model': 'gpt-4.1'}

    def get_provider(self) -> BaseProvider:
        LLM_PROVIDERS_DOCS = 'https://github.com/MeduzaAI/django-agent-framework/'

        provider_name = self.config.get('provider')

        if provider_name not in agent_settings.llm_providers:
            raise ImproperlyConfigured(
                'Unknown LLM provider %s. Please refer %s documentation on how to configure custom LLM Providers'
                % (provider_name, LLM_PROVIDERS_DOCS)
            )

        llm_provider = agent_settings.llm_providers[provider_name]
        default_llm_provider = agent_settings.llm_providers['default']
        backend_class_import_string = llm_provider.pop('backend') or default_llm_provider.get('backend')
        backend_class = perform_import(backend_class_import_string, f'llm_providers.{provider_name}.backend')
        return backend_class(**llm_provider)

    async def run(self, agent_input):
        messages = self.prepare_messages(agent_input)
        response = await self.generate(messages)
        if response.can_continue:
            return await self.run(response)
        return response

    def prepare_messages(self, agent_input):
        messages = self.memory

    async def get_messages(self):
        if not self.message_thread_id:
            pass

        messages = self.message_model.objects.filter(thread_id=self.message_thread_id)
