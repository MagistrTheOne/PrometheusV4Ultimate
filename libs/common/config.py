"""Configuration management for PrometheusULTIMATE v4."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class OrchestratorConfig(BaseModel):
    """Orchestrator configuration."""
    max_concurrent_tasks: int = 8
    task_timeout_s: int = 300
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {
        "max_retries": 2,
        "backoff": "linear"
    })


class PlannerConfig(BaseModel):
    """Planner configuration."""
    cost_weight: float = 0.5
    quality_weight: float = 0.5
    time_weight: float = 0.3
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {
        "max_retries": 2,
        "backoff": "linear"
    })


class CriticConfig(BaseModel):
    """Critic configuration."""
    fact_check: str = "simple"
    code_check: str = "run_tests"
    policy_check: str = "strict"
    auto_fix: bool = True


class MemoryConfig(BaseModel):
    """Memory configuration."""
    vector: Dict[str, Any] = Field(default_factory=lambda: {
        "top_k": 8,
        "reindex_ttl_h": 12,
        "collection_prefix": "promu_"
    })
    kv: Dict[str, Any] = Field(default_factory=lambda: {
        "connection_pool_size": 10,
        "query_timeout_s": 30
    })
    project_store: Dict[str, Any] = Field(default_factory=lambda: {
        "versioning": True,
        "max_versions": 10
    })


class SkillsConfig(BaseModel):
    """Skills configuration."""
    sandbox: Dict[str, Any] = Field(default_factory=lambda: {
        "cpu_ms": 800,
        "ram_mb": 512,
        "net": "off",
        "timeout_s": 30
    })
    registry: Dict[str, Any] = Field(default_factory=lambda: {
        "auto_reload": True,
        "validation": "strict"
    })


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    tracing: bool = True
    cost_accounting: bool = True
    metrics_retention_days: int = 30
    log_level: str = "INFO"


class SecurityConfig(BaseModel):
    """Security configuration."""
    pii_redaction: str = "basic"
    allow_external_providers: bool = False
    sandbox_strict: bool = True


class RoutingConfig(BaseModel):
    """Routing configuration."""
    policy: str = "auto"
    rules: list = Field(default_factory=list)
    fallback: list = Field(default_factory=lambda: ["radon/base-0.8b", "radon/small-0.1b"])
    disallow_external: bool = True


class Settings(BaseSettings):
    """Application settings."""
    
    # Environment
    promu_env: str = "dev"
    promu_port: int = 8090
    promu_data_dir: str = "./.promu"
    promu_secret_key: str = "dev-secret-key-change-in-production"
    
    # Vector Database
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    
    # KV Database
    sqlite_db_path: str = "./.promu/kv.db"
    
    # LLM Configuration
    disallow_external_providers: bool = True
    huggingface_token: str = ""
    model_registry_path: str = "./.promu/models"
    
    # Security & Sandbox
    sandbox_cpu_ms: int = 800
    sandbox_ram_mb: int = 512
    sandbox_net: str = "off"
    sandbox_timeout_s: int = 30
    
    # Observability
    tracing_enabled: bool = True
    cost_accounting: bool = True
    log_level: str = "INFO"
    
    # Service Ports
    gateway_port: int = 8090
    orchestrator_port: int = 8000
    planner_port: int = 8001
    critic_port: int = 8002
    memory_port: int = 8003
    skills_port: int = 8004
    observability_port: int = 8005
    ui_port: int = 3000
    
    # Component configurations
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def ensure_data_dir() -> Path:
    """Ensure data directory exists."""
    settings = get_settings()
    data_dir = Path(settings.promu_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
