"""prompts/__init__.py — expose loader at package level."""
from prompts.loader import load_prompt, load_prompt_metadata, list_available_prompts

__all__ = ["load_prompt", "load_prompt_metadata", "list_available_prompts"]
