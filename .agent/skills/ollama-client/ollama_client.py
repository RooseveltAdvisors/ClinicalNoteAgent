"""
Ollama Client Skill.

This module provides a wrapper for interacting with the Ollama API
for Phi-4 model inference with timeout handling and logging.
"""

import os
import requests
from typing import Optional, Dict, Any
from datetime import datetime
import time


class OllamaClient:
    """
    Client for Ollama API (Phi-4 model).

    Attributes:
        base_url: Ollama server URL (default: http://localhost:11434)
        model: Model name (default: phi4:14b)
        timeout: Request timeout in seconds (default: 300 = 5 minutes)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 300
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL (defaults to OLLAMA_HOST env var or localhost:11434)
            model: Model name (defaults to OLLAMA_MODEL env var or phi4:14b)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.model = model or os.getenv('OLLAMA_MODEL', 'phi4:14b')
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate text completion from Phi-4 model.

        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0-1.0, default: 0.1 for deterministic outputs)
            max_tokens: Maximum tokens to generate (default: None for model default)
            stop_sequences: Optional list of stop sequences

        Returns:
            Dict containing:
                - response: Generated text
                - execution_time_ms: Inference time in milliseconds
                - model: Model name used
                - success: Whether the request succeeded

        Raises:
            Exception: If Ollama API request fails after retries
        """
        start_time = time.time()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
            },
            "stream": False
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "response": result.get("response", ""),
                "execution_time_ms": execution_time_ms,
                "model": self.model,
                "success": True
            }

        except requests.exceptions.Timeout:
            execution_time_ms = int((time.time() - start_time) * 1000)
            raise Exception(
                f"Ollama request timed out after {self.timeout}s "
                f"(execution_time: {execution_time_ms}ms)"
            )

        except requests.exceptions.RequestException as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            raise Exception(
                f"Ollama request failed: {str(e)} "
                f"(execution_time: {execution_time_ms}ms)"
            )

    def check_health(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
