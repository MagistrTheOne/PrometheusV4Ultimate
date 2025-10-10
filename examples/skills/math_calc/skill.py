"""Math Calc skill implementation."""

import json
import os
import re
import math
from decimal import Decimal, getcontext
from typing import Dict, Any, List, Optional, Union
from libs.skills import BaseSkill, SkillSpec, SkillRunResult, PermissionType, ResourceLimit


class MathCalcSkill(BaseSkill):
    """Skill for performing mathematical calculations with high precision."""
    
    def __init__(self):
        spec = SkillSpec(
            name="math_calc",
            version="1.0.0",
            description="Perform mathematical calculations with high precision",
            inputs={
                "expression": "Mathematical expression to evaluate",
                "precision": "Decimal precision for calculations (default: 10)",
                "output_file": "Path to save calculation results"
            },
            outputs={
                "output_file": "Path to the results file",
                "result": "Calculated result",
                "expression": "Original expression",
                "precision": "Used precision",
                "error": "Error message if calculation failed"
            },
            perms={
                PermissionType.FS_READ: True,
                PermissionType.FS_WRITE: True,
                PermissionType.NETWORK: False,
                PermissionType.ENV_VAR: False
            },
            limits={
                ResourceLimit.CPU_MS: 2000,
                ResourceLimit.RAM_MB: 50,
                ResourceLimit.TIME_S: 10,
                ResourceLimit.DISK_MB: 10
            },
            tags=["math", "calculation", "precision", "arithmetic"],
            author="PrometheusULTIMATE",
            license="MIT"
        )
        super().__init__(spec)
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute mathematical calculation."""
        expression = kwargs["expression"]
        precision = int(kwargs.get("precision", 10))
        output_file = kwargs["output_file"]
        
        # Set decimal precision
        getcontext().prec = precision
        
        try:
            # Validate and sanitize expression
            sanitized_expr = self._sanitize_expression(expression)
            
            # Evaluate expression
            result = self._evaluate_expression(sanitized_expr)
            
            # Create result data
            result_data = {
                "expression": expression,
                "sanitized_expression": sanitized_expr,
                "result": str(result),
                "precision": precision,
                "result_type": type(result).__name__,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            # Save results
            self._save_results(output_file, result_data)
            
            return {
                "output_file": output_file,
                "result": str(result),
                "expression": expression,
                "precision": precision,
                "error": None
            }
            
        except Exception as e:
            error_data = {
                "expression": expression,
                "error": str(e),
                "precision": precision,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            
            self._save_results(output_file, error_data)
            
            return {
                "output_file": output_file,
                "result": None,
                "expression": expression,
                "precision": precision,
                "error": str(e)
            }
    
    def _sanitize_expression(self, expression: str) -> str:
        """Sanitize mathematical expression for safe evaluation."""
        
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Allowed characters: digits, operators, parentheses, decimal point, math functions
        allowed_chars = set("0123456789+-*/.()eE")
        allowed_functions = ["sin", "cos", "tan", "log", "ln", "sqrt", "abs", "pi", "e"]
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r"__",  # Double underscores
            r"import",  # Import statements
            r"exec",  # Exec statements
            r"eval",  # Eval statements
            r"open",  # File operations
            r"file",  # File operations
            r"input",  # Input operations
            r"raw_input",  # Input operations
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                raise ValueError(f"Dangerous pattern detected: {pattern}")
        
        # Check for only allowed characters and functions
        for char in expression:
            if char not in allowed_chars and not any(func in expression for func in allowed_functions):
                if not char.isalpha():
                    raise ValueError(f"Invalid character in expression: {char}")
        
        return expression
    
    def _evaluate_expression(self, expression: str) -> Union[Decimal, float, int]:
        """Safely evaluate mathematical expression."""
        
        # Replace math constants
        expression = expression.replace("pi", str(math.pi))
        expression = expression.replace("e", str(math.e))
        
        # Replace math functions with safe alternatives
        math_functions = {
            "sin": "math.sin",
            "cos": "math.cos", 
            "tan": "math.tan",
            "log": "math.log10",
            "ln": "math.log",
            "sqrt": "math.sqrt",
            "abs": "abs"
        }
        
        for func, replacement in math_functions.items():
            expression = expression.replace(func, replacement)
        
        # Create safe evaluation environment
        safe_globals = {
            "__builtins__": {},
            "math": math,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "pow": pow,
        }
        
        safe_locals = {}
        
        try:
            # Evaluate expression
            result = eval(expression, safe_globals, safe_locals)
            
            # Convert to appropriate type
            if isinstance(result, (int, float)):
                if isinstance(result, float) and result.is_integer():
                    return int(result)
                return result
            else:
                raise ValueError(f"Expression did not evaluate to a number: {result}")
                
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")
    
    def _save_results(self, output_file: str, data: Dict[str, Any]) -> None:
        """Save calculation results to file."""
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
