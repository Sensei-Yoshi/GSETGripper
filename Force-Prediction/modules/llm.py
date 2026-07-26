"""Compatibility imports for the relocated Gemini adapter.

New code should import from :mod:`modules.models.gemini` directly.
"""

from .models.gemini import GeminiClient, get_client, load_dotenv

__all__ = ["GeminiClient", "get_client", "load_dotenv"]
