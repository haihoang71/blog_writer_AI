"""config/__init__.py — expose key helpers at package level."""

from config.settings import Settings, get_settings
from config.constants import MAX_REVISIONS

__all__ = ["Settings", "get_settings", "MAX_REVISIONS"]
