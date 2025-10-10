"""Skills SDK types and protocols."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Union
from enum import Enum


class PermissionType(str, Enum):
    """Permission types for skills."""
    FS_READ = "fs_read"
    FS_WRITE = "fs_write"
    NETWORK = "network"
    ENV_VAR = "env_var"


class ResourceLimit(str, Enum):
    """Resource limit types."""
    CPU_MS = "cpu_ms"
    RAM_MB = "ram_mb"
    TIME_S = "time_s"
    DISK_MB = "disk_mb"


@dataclass
class SkillSpec:
    """Skill specification with metadata and constraints."""
    name: str
    version: str
    description: str
    inputs: Dict[str, str]  # input_name -> type_description
    outputs: Dict[str, str]  # output_name -> type_description
    perms: Dict[PermissionType, Union[bool, List[str]]]  # permission -> allowed_values
    limits: Dict[ResourceLimit, int]  # resource -> limit_value
    tags: List[str] = None
    author: str = "unknown"
    license: str = "MIT"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class SkillRunResult:
    """Result of skill execution."""
    success: bool
    outputs: Dict[str, Any]
    error: Optional[str] = None
    metrics: Dict[str, float] = None  # cpu_ms, ram_mb, latency_ms, etc.
    logs: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.logs is None:
            self.logs = []


class Skill(Protocol):
    """Base protocol for all skills."""
    
    @property
    def spec(self) -> SkillSpec:
        """Get skill specification."""
        ...
    
    def run(self, **kwargs) -> SkillRunResult:
        """Execute the skill with given inputs."""
        ...


class BaseSkill:
    """Base implementation of Skill protocol."""
    
    def __init__(self, spec: SkillSpec):
        self._spec = spec
    
    @property
    def spec(self) -> SkillSpec:
        """Get skill specification."""
        return self._spec
    
    def run(self, **kwargs) -> SkillRunResult:
        """Execute the skill with given inputs."""
        # This is a base implementation - subclasses should override
        return SkillRunResult(
            success=False,
            outputs={},
            error="BaseSkill.run() not implemented"
        )


@dataclass
class SandboxConfig:
    """Configuration for skill sandbox."""
    cpu_limit_ms: int = 1000
    ram_limit_mb: int = 100
    time_limit_s: int = 30
    disk_limit_mb: int = 50
    network_enabled: bool = False
    allowed_domains: List[str] = None
    read_only_fs: bool = True
    tmp_dir: str = "/tmp/skill_sandbox"
    
    def __post_init__(self):
        if self.allowed_domains is None:
            self.allowed_domains = []


@dataclass
class SandboxResult:
    """Result of sandbox execution."""
    exit_code: int
    stdout: str
    stderr: str
    metrics: Dict[str, float]
    logs: List[str]
    artifacts: List[str]  # paths to created files
