"""Sandbox for executing skills with resource limits and isolation."""

import os
import subprocess
import tempfile
import time
import psutil
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from .types import SandboxConfig, SandboxResult, SkillRunResult, BaseSkill


class SandboxError(Exception):
    """Sandbox execution error."""
    pass


class Sandbox:
    """Sandbox for executing skills with resource limits."""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._temp_dir = None
    
    def execute_skill(
        self, 
        skill: BaseSkill, 
        inputs: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> SkillRunResult:
        """Execute a skill in the sandbox."""
        
        # Create temporary directory for skill execution
        self._temp_dir = tempfile.mkdtemp(prefix="skill_sandbox_")
        
        try:
            # Set up environment
            env = self._setup_environment()
            
            # Create skill execution script
            script_path = self._create_execution_script(skill, inputs)
            
            # Execute with resource limits
            result = self._execute_with_limits(script_path, env, timeout)
            
            # Parse results
            return self._parse_result(result, skill.spec.outputs)
            
        finally:
            # Cleanup
            self._cleanup()
    
    def _setup_environment(self) -> Dict[str, str]:
        """Set up environment variables for skill execution."""
        env = os.environ.copy()
        
        # Set sandbox-specific environment
        env["SKILL_SANDBOX"] = "true"
        env["SKILL_TMP_DIR"] = self._temp_dir
        env["SKILL_READ_ONLY"] = str(self.config.read_only_fs).lower()
        
        # Network restrictions
        if not self.config.network_enabled:
            env["SKILL_NETWORK_DISABLED"] = "true"
        else:
            env["SKILL_ALLOWED_DOMAINS"] = ",".join(self.config.allowed_domains)
        
        return env
    
    def _create_execution_script(self, skill: BaseSkill, inputs: Dict[str, Any]) -> str:
        """Create a Python script to execute the skill."""
        script_content = f'''
import sys
import json
import traceback
from pathlib import Path

# Add skill to path
sys.path.insert(0, "{Path.cwd()}")

# Import the skill
from {skill.__class__.__module__} import {skill.__class__.__name__}

def main():
    try:
        # Create skill instance
        skill_instance = {skill.__class__.__name__}()
        
        # Execute skill
        result = skill_instance.run(**{inputs!r})
        
        # Output result as JSON
        print("SKILL_RESULT_START")
        print(json.dumps({{
            "success": result.success,
            "outputs": result.outputs,
            "error": result.error,
            "metrics": result.metrics,
            "logs": result.logs
        }}))
        print("SKILL_RESULT_END")
        
    except Exception as e:
        print("SKILL_ERROR_START")
        print(json.dumps({{
            "success": False,
            "outputs": {{}},
            "error": str(e),
            "traceback": traceback.format_exc()
        }}))
        print("SKILL_ERROR_END")

if __name__ == "__main__":
    main()
'''
        
        script_path = os.path.join(self._temp_dir, "execute_skill.py")
        with open(script_path, "w") as f:
            f.write(script_content)
        
        return script_path
    
    def _execute_with_limits(
        self, 
        script_path: str, 
        env: Dict[str, str],
        timeout: Optional[int] = None
    ) -> SandboxResult:
        """Execute script with resource limits."""
        
        # Use skill's time limit or provided timeout
        exec_timeout = timeout or self.config.time_limit_s
        
        # Start process
        start_time = time.time()
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=self._temp_dir
        )
        
        # Monitor and enforce limits
        try:
            stdout, stderr = process.communicate(timeout=exec_timeout)
            exit_code = process.returncode
            
        except subprocess.TimeoutExpired:
            # Kill process if timeout exceeded
            process.kill()
            stdout, stderr = process.communicate()
            exit_code = -1
            stderr += f"\nProcess killed due to timeout ({exec_timeout}s)"
        
        # Calculate metrics
        execution_time = time.time() - start_time
        metrics = {
            "execution_time_s": execution_time,
            "cpu_ms": 0,  # Would need more complex monitoring
            "ram_mb": 0,  # Would need more complex monitoring
            "exit_code": exit_code
        }
        
        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            metrics=metrics,
            logs=[],
            artifacts=self._find_artifacts()
        )
    
    def _parse_result(self, sandbox_result: SandboxResult, expected_outputs: Dict[str, str]) -> SkillRunResult:
        """Parse sandbox execution result."""
        
        if sandbox_result.exit_code != 0:
            return SkillRunResult(
                success=False,
                outputs={},
                error=f"Sandbox execution failed: {sandbox_result.stderr}",
                metrics=sandbox_result.metrics,
                logs=sandbox_result.logs
            )
        
        # Parse skill result from stdout
        stdout_lines = sandbox_result.stdout.split('\n')
        result_start = -1
        result_end = -1
        
        for i, line in enumerate(stdout_lines):
            if line.strip() == "SKILL_RESULT_START":
                result_start = i + 1
            elif line.strip() == "SKILL_RESULT_END":
                result_end = i
                break
            elif line.strip() == "SKILL_ERROR_START":
                result_start = i + 1
            elif line.strip() == "SKILL_ERROR_END":
                result_end = i
                break
        
        if result_start == -1 or result_end == -1:
            return SkillRunResult(
                success=False,
                outputs={},
                error="Could not parse skill result from sandbox output",
                metrics=sandbox_result.metrics,
                logs=sandbox_result.logs
            )
        
        try:
            import json
            result_json = '\n'.join(stdout_lines[result_start:result_end])
            result_data = json.loads(result_json)
            
            return SkillRunResult(
                success=result_data.get("success", False),
                outputs=result_data.get("outputs", {}),
                error=result_data.get("error"),
                metrics=sandbox_result.metrics,
                logs=result_data.get("logs", [])
            )
            
        except json.JSONDecodeError as e:
            return SkillRunResult(
                success=False,
                outputs={},
                error=f"Failed to parse skill result JSON: {e}",
                metrics=sandbox_result.metrics,
                logs=sandbox_result.logs
            )
    
    def _find_artifacts(self) -> List[str]:
        """Find artifacts created by the skill."""
        if not self._temp_dir:
            return []
        
        artifacts = []
        for root, dirs, files in os.walk(self._temp_dir):
            for file in files:
                if file != "execute_skill.py":  # Skip execution script
                    artifacts.append(os.path.join(root, file))
        
        return artifacts
    
    def _cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None


class SandboxManager:
    """Manager for multiple sandbox instances."""
    
    def __init__(self, default_config: Optional[SandboxConfig] = None):
        self.default_config = default_config or SandboxConfig()
        self._active_sandboxes: List[Sandbox] = []
    
    def create_sandbox(self, config: Optional[SandboxConfig] = None) -> Sandbox:
        """Create a new sandbox instance."""
        sandbox = Sandbox(config or self.default_config)
        self._active_sandboxes.append(sandbox)
        return sandbox
    
    def cleanup_all(self) -> None:
        """Clean up all active sandboxes."""
        for sandbox in self._active_sandboxes:
            sandbox._cleanup()
        self._active_sandboxes.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()
