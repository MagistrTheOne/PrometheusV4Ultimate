"""LLM providers for PrometheusULTIMATE v4."""

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .schemas import ChatMessage, ChatRole


class ChatResult:
    """Chat completion result."""
    
    def __init__(
        self,
        text: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        model: str = "",
        finish_reason: str = "stop"
    ):
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.model = model
        self.finish_reason = finish_reason


class ChatProvider(ABC):
    """Abstract chat provider interface."""
    
    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResult:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider health."""
        pass


class InternalStubProvider(ChatProvider):
    """Stub provider for testing and development."""
    
    def __init__(self):
        self.name = "internal_stub"
        self.models = [
            "radon/small-0.1b",
            "radon/base-0.8b", 
            "radon/balanced-3b",
            "radon/efficient-3b",
            "radon/ultra-13b",
            "radon/mega-70b",
            "oracle/moe-850b"
        ]
    
    async def chat(
        self,
        messages: List[ChatMessage],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResult:
        """Generate deterministic stub response."""
        
        # Create deterministic response based on input
        input_hash = hashlib.md5(
            f"{model}:{len(messages)}:{messages[-1].content if messages else ''}".encode()
        ).hexdigest()[:8]
        
        # Generate context-aware response
        last_message = messages[-1].content if messages else ""
        
        if "csv" in last_message.lower():
            response = f"[STUB] I'll help you process CSV data. Model: {model} (hash: {input_hash})"
        elif "code" in last_message.lower():
            response = f"[STUB] I'll analyze and fix the code. Model: {model} (hash: {input_hash})"
        elif "plot" in last_message.lower() or "graph" in last_message.lower():
            response = f"[STUB] I'll create a visualization. Model: {model} (hash: {input_hash})"
        elif "search" in last_message.lower() or "find" in last_message.lower():
            response = f"[STUB] I'll search the knowledge base. Model: {model} (hash: {input_hash})"
        else:
            response = f"[STUB] I understand your request. Model: {model} (hash: {input_hash})"
        
        # Simulate token usage
        tokens_in = sum(len(msg.content.split()) for msg in messages)
        tokens_out = len(response.split())
        
        return ChatResult(
            text=response,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            finish_reason="stop"
        )
    
    async def health_check(self) -> bool:
        """Always healthy."""
        return True


class ExternalProvider(ChatProvider):
    """External provider (disabled in current version)."""
    
    def __init__(self, name: str, base_url: str, api_key: str):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
    
    async def chat(
        self,
        messages: List[ChatMessage],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResult:
        """External provider is disabled."""
        raise Exception("External providers are disabled. Use internal models only.")
    
    async def health_check(self) -> bool:
        """External provider is disabled."""
        return False


class ProviderRegistry:
    """Provider registry and routing."""
    
    def __init__(self):
        self.providers: Dict[str, ChatProvider] = {}
        self.default_provider = InternalStubProvider()
        self._register_default_providers()
    
    def _register_default_providers(self):
        """Register default providers."""
        self.providers["internal_stub"] = self.default_provider
    
    def register_provider(self, name: str, provider: ChatProvider):
        """Register a new provider."""
        self.providers[name] = provider
    
    def get_provider(self, name: str) -> ChatProvider:
        """Get provider by name."""
        return self.providers.get(name, self.default_provider)
    
    def list_providers(self) -> List[str]:
        """List available providers."""
        return list(self.providers.keys())
    
    async def route_request(
        self,
        messages: List[ChatMessage],
        *,
        model: str,
        provider: Optional[str] = None,
        **kwargs
    ) -> ChatResult:
        """Route request to appropriate provider."""
        
        # Determine provider based on model or explicit choice
        if provider:
            target_provider = self.get_provider(provider)
        else:
            # Route based on model family
            if model.startswith("radon/") or model.startswith("oracle/"):
                target_provider = self.default_provider
            else:
                target_provider = self.default_provider
        
        return await target_provider.chat(
            messages=messages,
            model=model,
            **kwargs
        )


# Global provider registry
provider_registry = ProviderRegistry()
