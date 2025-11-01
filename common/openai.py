"""OpenAI client for RE module - standalone implementation"""
from __future__ import annotations
import os
import logging
from typing import Any, Optional
from collections.abc import AsyncIterator

try:
    import pipmaster as pm
    if not pm.is_installed("openai"):
        pm.install("openai")
except ImportError:
    pass

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger("lightrag.re")


class InvalidResponseError(Exception):
    """Custom exception for invalid responses"""
    pass


def create_openai_async_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client with configuration.
    
    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        base_url: Base URL for API. If None, uses OPENAI_API_BASE or default.
        **kwargs: Additional client configuration.
    
    Returns:
        AsyncOpenAI client instance
    """
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("API key required (OPENAI_API_KEY env var or api_key parameter)")
    
    if base_url is None:
        base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    # Default headers
    default_headers = {
        "User-Agent": "RE/1.0",
        "Content-Type": "application/json",
    }
    
    # Merge configurations
    config = {
        "api_key": api_key,
        "base_url": base_url,
        "default_headers": {**default_headers, **kwargs.get("default_headers", {})},
    }
    
    # Add any additional config
    for key, value in kwargs.items():
        if key not in ["default_headers"]:
            config[key] = value
    
    return AsyncOpenAI(**config)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
async def complete(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 180,
    **kwargs
) -> str:
    """Complete a prompt using OpenAI API.
    
    Args:
        model: Model name (e.g., "gpt-4", "gpt-4o", "qwen")
        prompt: User prompt
        system_prompt: Optional system prompt
        history_messages: Optional conversation history
        api_key: API key (uses env var if not provided)
        base_url: API base URL (uses env var if not provided)
        timeout: Request timeout in seconds
        **kwargs: Additional parameters for chat.completions.create
    
    Returns:
        Response content string
    """
    # Create client
    client = create_openai_async_client(api_key=api_key, base_url=base_url)
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if history_messages:
        messages.extend(history_messages)
    else:
        messages.append({"role": "user", "content": prompt})
    
    try:
        # Make API call
        # ModelScope and some other APIs require enable_thinking=false for non-streaming calls
        # Use extra_body to pass custom parameters that aren't in the standard OpenAI SDK
        
        # Check if we need to add enable_thinking via extra_body for ModelScope API
        base_url_str = base_url or os.environ.get("OPENAI_API_BASE", "")
        if "modelscope" in base_url_str.lower():
            # ModelScope requires enable_thinking in the request body
            if "extra_body" not in kwargs:
                kwargs["extra_body"] = {}
            if "enable_thinking" not in kwargs.get("extra_body", {}):
                kwargs["extra_body"]["enable_thinking"] = False
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
            **kwargs
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
    ),
)
async def embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 30,
    **kwargs
) -> list[list[float]]:
    """Get embeddings for texts.
    
    Args:
        texts: List of texts to embed
        model: Embedding model name
        api_key: API key (uses env var if not provided)
        base_url: API base URL (uses env var if not provided)
        timeout: Request timeout in seconds
        **kwargs: Additional parameters
    
    Returns:
        List of embedding vectors
    """
    # Create client
    client = create_openai_async_client(api_key=api_key, base_url=base_url)
    
    try:
        response = await client.embeddings.create(
            model=model,
            input=texts,
            timeout=timeout,
            **kwargs
        )
        
        return [item.embedding for item in response.data]
        
    except Exception as e:
        logger.error(f"OpenAI embedding failed: {e}")
        raise


# Convenience wrapper class
class OpenAIRE:
    """Convenience wrapper for OpenAI API in RE module"""
    
    def __init__(
        self,
        model: str = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.api_key = api_key
        self.base_url = base_url
        self.embedding_model = embedding_model
        self._client = None
    
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of client"""
        if self._client is None:
            self._client = create_openai_async_client(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        **kwargs
    ) -> str:
        """Complete a prompt"""
        return await complete(
            model=self.model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs
        )
    
    async def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Get embeddings"""
        return await embed(
            texts=texts,
            model=self.embedding_model,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs
        )
