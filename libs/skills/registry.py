"""Skills registry and management."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from .sdk import BaseSkill, registry
from .types import SkillSpec


class SkillRegistryManager:
    """Manager for skill registry operations."""
    
    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path) if registry_path else Path(".promu/skills")
        self.registry_path.mkdir(parents=True, exist_ok=True)
    
    def register_skill_from_path(self, skill_path: str) -> bool:
        """Register a skill from a directory path."""
        skill_path = Path(skill_path)
        
        if not skill_path.exists():
            raise FileNotFoundError(f"Skill path does not exist: {skill_path}")
        
        # Look for spec.yaml
        spec_file = skill_path / "spec.yaml"
        if not spec_file.exists():
            raise FileNotFoundError(f"spec.yaml not found in {skill_path}")
        
        # Load spec
        with open(spec_file, "r") as f:
            spec_data = yaml.safe_load(f)
        
        # Create skill spec
        spec = SkillSpec(
            name=spec_data["name"],
            version=spec_data["version"],
            description=spec_data.get("description", ""),
            inputs=spec_data.get("inputs", {}),
            outputs=spec_data.get("outputs", {}),
            perms=spec_data.get("permissions", {}),
            limits=spec_data.get("limits", {}),
            tags=spec_data.get("tags", []),
            author=spec_data.get("author", "unknown"),
            license=spec_data.get("license", "MIT")
        )
        
        # Import skill module
        skill_module_path = skill_path / "skill.py"
        if not skill_module_path.exists():
            raise FileNotFoundError(f"skill.py not found in {skill_path}")
        
        # Add to Python path and import
        import sys
        import importlib.util
        
        spec_module = importlib.util.spec_from_file_location(
            f"skill_{spec.name}_{spec.version}", 
            skill_module_path
        )
        skill_module = importlib.util.module_from_spec(spec_module)
        sys.modules[f"skill_{spec.name}_{spec.version}"] = skill_module
        spec_module.loader.exec_module(skill_module)
        
        # Find skill class (should be the main class in the module)
        skill_class = None
        for attr_name in dir(skill_module):
            attr = getattr(skill_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BaseSkill) and 
                attr != BaseSkill):
                skill_class = attr
                break
        
        if not skill_class:
            raise ValueError(f"No skill class found in {skill_module_path}")
        
        # Register skill
        registry.register(skill_class)
        
        # Save registry info
        self._save_skill_info(spec, skill_path)
        
        return True
    
    def list_registered_skills(self) -> List[Dict[str, Any]]:
        """List all registered skills."""
        return registry.list_skills()
    
    def get_skill_info(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed info about a skill."""
        return registry.get_skill_info(name, version)
    
    def validate_skill(self, skill_path: str) -> Dict[str, Any]:
        """Validate a skill without registering it."""
        skill_path = Path(skill_path)
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "spec": None
        }
        
        try:
            # Check required files
            if not (skill_path / "spec.yaml").exists():
                validation_result["errors"].append("spec.yaml not found")
                validation_result["valid"] = False
            
            if not (skill_path / "skill.py").exists():
                validation_result["errors"].append("skill.py not found")
                validation_result["valid"] = False
            
            if not (skill_path / "tests").exists():
                validation_result["warnings"].append("tests directory not found")
            
            # Validate spec.yaml
            if (skill_path / "spec.yaml").exists():
                with open(skill_path / "spec.yaml", "r") as f:
                    spec_data = yaml.safe_load(f)
                
                # Check required fields
                required_fields = ["name", "version", "description"]
                for field in required_fields:
                    if field not in spec_data:
                        validation_result["errors"].append(f"Missing required field: {field}")
                        validation_result["valid"] = False
                
                validation_result["spec"] = spec_data
            
            # Validate skill.py syntax
            if (skill_path / "skill.py").exists():
                try:
                    with open(skill_path / "skill.py", "r") as f:
                        compile(f.read(), str(skill_path / "skill.py"), "exec")
                except SyntaxError as e:
                    validation_result["errors"].append(f"Syntax error in skill.py: {e}")
                    validation_result["valid"] = False
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
            validation_result["valid"] = False
        
        return validation_result
    
    def _save_skill_info(self, spec: SkillSpec, skill_path: Path) -> None:
        """Save skill information to registry."""
        registry_file = self.registry_path / "registry.json"
        
        # Load existing registry
        registry_data = {}
        if registry_file.exists():
            with open(registry_file, "r") as f:
                registry_data = json.load(f)
        
        # Add skill info
        skill_key = f"{spec.name}@{spec.version}"
        registry_data[skill_key] = {
            "name": spec.name,
            "version": spec.version,
            "description": spec.description,
            "author": spec.author,
            "license": spec.license,
            "tags": spec.tags,
            "path": str(skill_path),
            "inputs": spec.inputs,
            "outputs": spec.outputs,
            "permissions": spec.perms,
            "limits": spec.limits,
        }
        
        # Save registry
        with open(registry_file, "w") as f:
            json.dump(registry_data, f, indent=2)
    
    def load_registry(self) -> None:
        """Load skills from registry file."""
        registry_file = self.registry_path / "registry.json"
        
        if not registry_file.exists():
            return
        
        with open(registry_file, "r") as f:
            registry_data = json.load(f)
        
        # Register all skills from registry
        for skill_key, skill_info in registry_data.items():
            skill_path = skill_info.get("path")
            if skill_path and Path(skill_path).exists():
                try:
                    self.register_skill_from_path(skill_path)
                except Exception as e:
                    print(f"Warning: Failed to load skill {skill_key}: {e}")
    
    def unregister_skill(self, name: str, version: str) -> bool:
        """Unregister a skill."""
        # This would require modifying the global registry
        # For now, just remove from registry file
        registry_file = self.registry_path / "registry.json"
        
        if not registry_file.exists():
            return False
        
        with open(registry_file, "r") as f:
            registry_data = json.load(f)
        
        skill_key = f"{name}@{version}"
        if skill_key in registry_data:
            del registry_data[skill_key]
            
            with open(registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)
            
            return True
        
        return False
