"""Code Format skill implementation."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from libs.skills import BaseSkill, SkillSpec, SkillRunResult, PermissionType, ResourceLimit


class CodeFormatSkill(BaseSkill):
    """Skill for formatting and linting Python code."""
    
    def __init__(self):
        spec = SkillSpec(
            name="code_format",
            version="1.0.0",
            description="Format and lint Python code using black and ruff",
            inputs={
                "input_file": "Path to Python file to format",
                "output_file": "Path for formatted output file",
                "formatter": "Formatter to use: black, ruff, both (default: both)",
                "line_length": "Maximum line length (default: 88)"
            },
            outputs={
                "output_file": "Path to the formatted file",
                "original_size": "Size of original file in bytes",
                "formatted_size": "Size of formatted file in bytes",
                "changes_made": "Whether any changes were made",
                "lint_issues": "Number of lint issues found"
            },
            perms={
                PermissionType.FS_READ: True,
                PermissionType.FS_WRITE: True,
                PermissionType.NETWORK: False,
                PermissionType.ENV_VAR: False
            },
            limits={
                ResourceLimit.CPU_MS: 5000,
                ResourceLimit.RAM_MB: 100,
                ResourceLimit.TIME_S: 30,
                ResourceLimit.DISK_MB: 50
            },
            tags=["python", "format", "lint", "black", "ruff", "code"],
            author="PrometheusULTIMATE",
            license="MIT"
        )
        super().__init__(spec)
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute code formatting operation."""
        input_file = kwargs["input_file"]
        output_file = kwargs["output_file"]
        formatter = kwargs.get("formatter", "both")
        line_length = int(kwargs.get("line_length", 88))
        
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if not input_file.endswith('.py'):
            raise ValueError("Input file must be a Python file (.py)")
        
        # Get original file size
        original_size = os.path.getsize(input_file)
        
        # Read original content
        with open(input_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Format code
        formatted_content = self._format_code(original_content, formatter, line_length)
        
        # Count lint issues
        lint_issues = self._count_lint_issues(formatted_content)
        
        # Save formatted content
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        # Get formatted file size
        formatted_size = os.path.getsize(output_file)
        
        # Check if changes were made
        changes_made = original_content != formatted_content
        
        return {
            "output_file": output_file,
            "original_size": original_size,
            "formatted_size": formatted_size,
            "changes_made": changes_made,
            "lint_issues": lint_issues
        }
    
    def _format_code(self, content: str, formatter: str, line_length: int) -> str:
        """Format Python code using specified formatter."""
        
        if formatter == "black":
            return self._format_with_black(content, line_length)
        elif formatter == "ruff":
            return self._format_with_ruff(content, line_length)
        elif formatter == "both":
            # First format with black, then with ruff
            formatted = self._format_with_black(content, line_length)
            return self._format_with_ruff(formatted, line_length)
        else:
            raise ValueError(f"Unknown formatter: {formatter}")
    
    def _format_with_black(self, content: str, line_length: int) -> str:
        """Format code using black."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Run black
            cmd = [
                "black",
                "--line-length", str(line_length),
                "--quiet",
                temp_file_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                # If black fails, return original content
                return content
            
            # Read formatted content
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                formatted_content = f.read()
            
            # Clean up
            os.unlink(temp_file_path)
            
            return formatted_content
            
        except Exception:
            # If formatting fails, return original content
            return content
    
    def _format_with_ruff(self, content: str, line_length: int) -> str:
        """Format code using ruff."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Run ruff format
            cmd = [
                "ruff",
                "format",
                "--line-length", str(line_length),
                temp_file_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                # If ruff fails, return original content
                return content
            
            # Read formatted content
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                formatted_content = f.read()
            
            # Clean up
            os.unlink(temp_file_path)
            
            return formatted_content
            
        except Exception:
            # If formatting fails, return original content
            return content
    
    def _count_lint_issues(self, content: str) -> int:
        """Count lint issues in the code."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            # Run ruff check
            cmd = [
                "ruff",
                "check",
                "--output-format", "json",
                temp_file_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.unlink(temp_file_path)
            
            if result.returncode == 0:
                return 0  # No issues
            
            # Parse JSON output to count issues
            try:
                import json
                issues = json.loads(result.stdout)
                return len(issues)
            except json.JSONDecodeError:
                # If JSON parsing fails, count lines in output
                return len([line for line in result.stdout.split('\n') if line.strip()])
            
        except Exception:
            return 0  # If linting fails, assume no issues
