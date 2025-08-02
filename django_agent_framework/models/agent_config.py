from django.db import models


class AgentConfig(models.Model):
    name = models.CharField(max_length=255, unique=True)
    tools = models.JSONField(default=list)

    model = models.CharField(max_length=255)
    provider = models.CharField(max_length=255)
    model_settings = models.JSONField(default=dict)

    memory_config = models.JSONField(default=dict)
