"""
Settings for Django Agent framework are all namespaced in the DJANGO_AGENT_FRAMEWORK setting.
For example your project's `settings.py` file might look like this:

DJANGO_AGENT_FRAMEWORK = {
    'llm_providers': {
        'grok': {
            'backend': 'django_agent_framework.llm_providers.openai.OpenAILLMProvider',
            'api_key': os.getenv('GROK_API_KEY'),
            'base_url': 'https://api.grok.com'
        },
    }
}

This module provides the `api_setting` object, that is used to access
Agent framework settings, checking for user settings first, then falling
back to the defaults.
"""

from django.conf import settings
# Import from `django.core.signals` instead of the official location
# `django.test.signals` to avoid importing the test module unnecessarily.
from django.core.signals import setting_changed
from django.utils.module_loading import import_string



DEFAULTS = {
    'llm_providers': {
        'default': {
            'backend': 'django_agent_framework.llm_providers.openai.OpenAILLMProvider',
        }
    }
}

def perform_import(val, setting_name):
    """
    If the given setting is a string import notation,
    then perform the necessary import or imports.
    """
    if val is None:
        return None
    elif isinstance(val, str):
        return import_from_string(val, setting_name)
    elif isinstance(val, (list, tuple)):
        return [import_from_string(item, setting_name) for item in val]
    return val


def import_from_string(val, setting_name):
    """
    Attempt to import a class from a string representation.
    """
    try:
        return import_string(val)
    except ImportError as e:
        msg = "Could not import '%s' for Agent setting '%s'. %s: %s." % (val, setting_name, e.__class__.__name__, e)
        raise ImportError(msg)


class AgentFrameworkSettings:
    def __init__(self, user_settings=None, defaults=None):
        self._user_settings = user_settings or {}
        self.defaults = defaults or DEFAULTS
        self._cached_attrs = set()

    @property
    def user_settings(self):
        if not hasattr(self, '_user_settings'):
            self._user_settings = getattr(settings, 'DJANGO_AGENT_FRAMEWORK', {})
        return self._user_settings

    def __getattr__(self, attr):
        if attr not in self.defaults:
            raise AttributeError("Invalid Agent setting: '%s'" % attr)

        try:
            val = self.user_settings[attr]
        except KeyError:
            val = self.defaults[attr]

        self._cached_attrs.add(attr)
        setattr(self, attr, val)
        return val

agent_settings = AgentFrameworkSettings(None, DEFAULTS)

def reload_api_settings(*args, **kwargs):
    setting = kwargs['setting']
    if setting == 'DJANGO_AGENT_FRAMEWORK':
        agent_settings.reload()


setting_changed.connect(reload_api_settings)