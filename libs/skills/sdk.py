"""Skills SDK base classes and decorators."""

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from .types import Skill, SkillRunResult, SkillSpec, PermissionType, ResourceLimit


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
        start_time = time.time()
        logs = []
        
        try:
            # Validate inputs
            self._validate_inputs(kwargs)
            
            # Execute skill logic
            outputs = self._execute(**kwargs)
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            metrics = {
                "latency_ms": latency_ms,
                "cpu_ms": 0,  # Will be filled by sandbox
                "ram_mb": 0,  # Will be filled by sandbox
            }
            
            return SkillRunResult(
                success=True,
                outputs=outputs,
                metrics=metrics,
                logs=logs
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return SkillRunResult(
                success=False,
                outputs={},
                error=str(e),
                metrics={"latency_ms": latency_ms},
                logs=logs
            )
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters against spec."""
        for input_name, input_type in self.spec.inputs.items():
            if input_name not in inputs:
                raise ValueError(f"Missing required input: {input_name}")
            # Basic type validation could be added here
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement _execute method")


def skill(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    inputs: Optional[Dict[str, str]] = None,
    outputs: Optional[Dict[str, str]] = None,
    perms: Optional[Dict[PermissionType, Union[bool, List[str]]]] = None,
    limits: Optional[Dict[ResourceLimit, int]] = None,
    tags: Optional[List[str]] = None,
    author: str = "unknown",
    license: str = "MIT"
):
    """Decorator to create a skill from a function."""
    
    if inputs is None:
        inputs = {}
    if outputs is None:
        outputs = {}
    if perms is None:
        perms = {}
    if limits is None:
        limits = {}
    if tags is None:
        tags = []
    
    def decorator(func: Callable) -> Type[BaseSkill]:
        """Create a skill class from a function."""
        
        class FunctionSkill(BaseSkill):
            """Skill created from a function."""
            
            def __init__(self):
                spec = SkillSpec(
                    name=name,
                    version=version,
                    description=description,
                    inputs=inputs,
                    outputs=outputs,
                    perms=perms,
                    limits=limits,
                    tags=tags,
                    author=author,
                    license=license
                )
                super().__init__(spec)
                self._func = func
            
            def _execute(self, **kwargs) -> Dict[str, Any]:
                """Execute the wrapped function."""
                result = self._func(**kwargs)
                
                # If function returns a dict, use it as outputs
                if isinstance(result, dict):
                    return result
                
                # Otherwise, wrap in a single output
                return {"result": result}
        
        return FunctionSkill
    
    return decorator


class SkillRegistry:
    """Registry for managing skills."""
    
    def __init__(self):
        self._skills: Dict[str, Type[BaseSkill]] = {}
        self._versions: Dict[str, List[str]] = {}
    
    def register(self, skill_class: Type[BaseSkill]) -> None:
        """Register a skill class."""
        spec = skill_class().spec
        skill_key = f"{spec.name}@{spec.version}"
        
        self._skills[skill_key] = skill_class
        
        if spec.name not in self._versions:
            self._versions[spec.name] = []
        self._versions[spec.name].append(spec.version)
        self._versions[spec.name].sort(reverse=True)  # Latest first
    
    def get_skill(self, name: str, version: Optional[str] = None) -> Optional[Type[BaseSkill]]:
        """Get a skill by name and optional version."""
        if version:
            skill_key = f"{name}@{version}"
            return self._skills.get(skill_key)
        
        # Get latest version
        if name in self._versions and self._versions[name]:
            latest_version = self._versions[name][0]
            skill_key = f"{name}@{latest_version}"
            return self._skills.get(skill_key)
        
        return None
    
    def list_skills(self) -> List[Dict[str, Any]]:
        """List all registered skills."""
        result = []
        for skill_key, skill_class in self._skills.items():
            spec = skill_class().spec
            result.append({
                "name": spec.name,
                "version": spec.version,
                "description": spec.description,
                "author": spec.author,
                "tags": spec.tags,
                "inputs": list(spec.inputs.keys()),
                "outputs": list(spec.outputs.keys()),
            })
        return result
    
    def get_skill_info(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed info about a skill."""
        skill_class = self.get_skill(name, version)
        if not skill_class:
            return None
        
        spec = skill_class().spec
        return {
            "name": spec.name,
            "version": spec.version,
            "description": spec.description,
            "author": spec.author,
            "license": spec.license,
            "tags": spec.tags,
            "inputs": spec.inputs,
            "outputs": spec.outputs,
            "permissions": spec.perms,
            "limits": spec.limits,
        }


# Global registry instance
registry = SkillRegistry()
