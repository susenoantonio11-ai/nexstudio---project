"""
NEX Studio · Gemini Provider
============================
Adapter untuk Google Gemini API yang sesuai dengan provider interface backend.
Pakai google-generativeai SDK official. Mendukung streaming, multi-turn,
plus system instructions.

Env vars yang dipakai yaitu GEMINI_API_KEY (wajib), GEMINI_MODEL (opsional,
default gemini-2.0-flash), GEMINI_TEMPERATURE (opsional, default 0.7).

Referensi resmi · https://ai.google.dev/gemini-api/docs
"""
import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Run: pip install google-generativeai")


class GeminiProvider:
    """Provider class untuk Gemini API · stateless, thread-safe."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai library tidak terinstall")

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY env var atau parameter wajib di-set")

        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"Gemini provider initialized · model={self.model_name}")

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text response · synchronous, satu turn."""
        config = {
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if max_tokens:
            config["max_output_tokens"] = max_tokens

        full_prompt = prompt
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{prompt}"

        try:
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(**config)
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generate failed · {str(e)}")
            raise

    def generate_structured(
        self,
        prompt: str,
        schema_hint: str = "JSON",
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate dengan instruksi structured output (JSON)."""
        structured_hint = f"\n\nRespond ONLY in valid {schema_hint} format. No markdown fences, no commentary."
        full_prompt = prompt + structured_hint
        return self.generate(full_prompt, system_instruction=system_instruction)

    def health_check(self) -> Dict[str, Any]:
        """Check apakah Gemini API reachable plus model available."""
        try:
            response = self.model.generate_content("ping")
            return {
                "status": "ok",
                "provider": "gemini",
                "model": self.model_name,
                "response_length": len(response.text) if response.text else 0
            }
        except Exception as e:
            return {
                "status": "error",
                "provider": "gemini",
                "model": self.model_name,
                "error": str(e)
            }


_provider_instance: Optional[GeminiProvider] = None


def get_gemini_provider() -> GeminiProvider:
    """Singleton accessor · reuse instance untuk efisiensi."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = GeminiProvider()
    return _provider_instance
