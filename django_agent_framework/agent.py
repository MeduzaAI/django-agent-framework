from django.core.exceptions import ImproperlyConfigured

from django_agent_framework.settings import agent_settings, perform_import


class Agent:
    def _get_llm_provider(self, name: str):
        LLM_PROVIDERS_DOCS = 'https://github.com/MeduzaAI/django-agent-framework/'

        if name not in agent_settings.llm_providers:
            raise ImproperlyConfigured(
                'Unknown LLM provider %s. Please refer %s documentation on how to configure custom LLM Providers' % (name, LLM_PROVIDERS_DOCS)
            )

        llm_provider = agent_settings.llm_providers[name]
        default_llm_provider = agent_settings.llm_providers['default']
        backend_class_import_string = llm_provider.pop('backend') or default_llm_provider.get('backend')
        backend_class = perform_import(backend_class_import_string, f'llm_providers.{name}.backend')
        return backend_class(**llm_provider)