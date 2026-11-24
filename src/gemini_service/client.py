"""
Gemini API Client Wrapper for CrowdSafe AI.
Wraps the official google-genai SDK, handles environment variable authentication,
rate limits, timeouts, and graceful offline fallbacks.
"""

import os
import io
import time
import logging
from typing import Optional, List, Dict, Any, Union
from PIL import Image

logger = logging.getLogger("GeminiClient")


class GeminiClient:
    """
    Manages connections and calls to Google Gemini API using the official google-genai SDK.
    Never exposes API keys to client code and provides seamless offline fallback.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        self.text_model = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
        self.vision_model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")
        self.timeout = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "25"))
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the official Google GenAI client if API key is present."""
        if not self.api_key:
            logger.info("No GEMINI_API_KEY found. Operating in local fallback mode.")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            logger.info("Google Gemini Client initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize Google Gemini Client: {e}")
            self._client = None

    def is_available(self) -> bool:
        """Returns True if the Gemini client is initialized with an API key."""
        return self._client is not None and bool(self.api_key)

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Generate text response using Gemini API.
        Returns dict with keys: 'text', 'success', 'fallback_needed', 'error'.
        """
        if not self.is_available():
            return {
                "text": "",
                "success": False,
                "fallback_needed": True,
                "error": "Gemini API key not configured or client unavailable."
            }

        target_model = model or self.text_model

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction
            )
            response = self._client.models.generate_content(
                model=target_model,
                contents=prompt,
                config=config
            )

            text_output = response.text if response and response.text else ""
            return {
                "text": text_output.strip(),
                "success": True,
                "fallback_needed": False,
                "error": None
            }
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Gemini generate_text error: {err_msg}")
            is_quota = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
            return {
                "text": "",
                "success": False,
                "fallback_needed": True,
                "error": f"Gemini API error ({'Quota Exhausted' if is_quota else 'General'}): {err_msg}"
            }

    def generate_multimodal(
        self,
        prompt: str,
        image: Union[bytes, Image.Image],
        mime_type: str = "image/jpeg",
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Generate analysis using Gemini Multimodal Vision API.
        Takes prompt and image (raw JPEG/PNG bytes or PIL Image).
        """
        if not self.is_available():
            return {
                "text": "",
                "success": False,
                "fallback_needed": True,
                "error": "Gemini API key not configured or client unavailable."
            }

        target_model = model or self.vision_model

        try:
            from google.genai import types

            if isinstance(image, bytes):
                part = types.Part.from_bytes(data=image, mime_type=mime_type)
            elif isinstance(image, Image.Image):
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
            else:
                return {
                    "text": "",
                    "success": False,
                    "fallback_needed": True,
                    "error": "Invalid image type. Expected bytes or PIL.Image."
                }

            config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_instruction
            )

            response = self._client.models.generate_content(
                model=target_model,
                contents=[part, prompt],
                config=config
            )

            text_output = response.text if response and response.text else ""
            return {
                "text": text_output.strip(),
                "success": True,
                "fallback_needed": False,
                "error": None
            }
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Gemini generate_multimodal error: {err_msg}")
            is_quota = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
            return {
                "text": "",
                "success": False,
                "fallback_needed": True,
                "error": f"Gemini Vision API error ({'Quota Exhausted' if is_quota else 'General'}): {err_msg}"
            }

    def create_chat(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        model: Optional[str] = None
    ) -> Optional[Any]:
        """Create a multi-turn chat session with optional function tools."""
        if not self.is_available():
            return None

        target_model = model or self.text_model
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=system_instruction,
                tools=tools
            )
            chat = self._client.chats.create(model=target_model, config=config)
            return chat
        except Exception as e:
            logger.warning(f"Failed to create Gemini chat session: {e}")
            return None

