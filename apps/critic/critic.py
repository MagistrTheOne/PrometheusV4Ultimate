"""Critic service for validation and quality control."""

import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from libs.common.llm_providers import provider_registry, ChatMessage, ChatRole
from libs.memory.vector_store import vector_store


class ValidationResult:
    """Result of validation check."""
    
    def __init__(
        self,
        passed: bool,
        issues: List[str],
        suggestions: List[str],
        confidence: float = 1.0
    ):
        self.passed = passed
        self.issues = issues
        self.suggestions = suggestions
        self.confidence = confidence


class FactChecker:
    """Fact checking using Memory and LLM."""
    
    def __init__(self):
        self.vector_store = vector_store
    
    async def check_facts(
        self,
        content: str,
        project_id: str
    ) -> ValidationResult:
        """Check facts against Memory."""
        
        # Extract claims from content
        claims = self._extract_claims(content)
        
        issues = []
        suggestions = []
        
        for claim in claims:
            # Search Memory for supporting evidence
            evidence = await self._search_evidence(claim, project_id)
            
            if not evidence:
                issues.append(f"No evidence found for claim: {claim}")
                suggestions.append(f"Verify claim: {claim}")
            else:
                # Check consistency with evidence
                consistency = await self._check_consistency(claim, evidence)
                if consistency < 0.7:
                    issues.append(f"Claim inconsistent with evidence: {claim}")
                    suggestions.append(f"Review claim against evidence: {claim}")
        
        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            confidence=0.8 if len(issues) == 0 else 0.3
        )
    
    def _extract_claims(self, content: str) -> List[str]:
        """Extract factual claims from content."""
        # Simple heuristic: sentences with numbers or specific facts
        sentences = re.split(r'[.!?]+', content)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Look for factual patterns
            if any(pattern in sentence.lower() for pattern in [
                'is', 'are', 'was', 'were', 'has', 'have', 'will', 'can', 'should'
            ]) and any(char.isdigit() for char in sentence):
                claims.append(sentence)
        
        return claims[:5]  # Limit to 5 claims
    
    async def _search_evidence(self, claim: str, project_id: str) -> List[Dict[str, Any]]:
        """Search for evidence in Memory."""
        try:
            # Create a simple embedding vector (placeholder)
            # In real implementation, would use proper embedding model
            query_vector = [0.1] * 384  # Placeholder vector
            
            results = await self.vector_store.search(
                project_id=project_id,
                query_vector=query_vector,
                top_k=3
            )
            
            return [result[2] for result in results]  # Return payloads
        except Exception:
            return []
    
    async def _check_consistency(self, claim: str, evidence: List[Dict[str, Any]]) -> float:
        """Check consistency between claim and evidence."""
        if not evidence:
            return 0.0
        
        # Simple consistency check using LLM
        messages = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="You are a fact checker. Rate consistency between a claim and evidence on a scale of 0-1."
            ),
            ChatMessage(
                role=ChatRole.USER,
                content=f"""
Claim: {claim}

Evidence: {evidence}

Rate consistency from 0 (completely inconsistent) to 1 (completely consistent).
Return only a number between 0 and 1.
"""
            )
        ]
        
        try:
            response = await provider_registry.route_request(
                messages=messages,
                model="radon/balanced-3b",
                temperature=0.1
            )
            
            # Extract number from response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', response.text)
            if numbers:
                return float(numbers[0])
            
        except Exception:
            pass
        
        return 0.5  # Default neutral score


class NumberChecker:
    """Number validation and recalculation."""
    
    def __init__(self):
        self.tolerance = 0.01  # 1% tolerance
    
    async def check_numbers(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Check numbers for accuracy."""
        
        # Extract numbers and calculations
        numbers = self._extract_numbers(content)
        calculations = self._extract_calculations(content)
        
        issues = []
        suggestions = []
        
        # Check calculations
        for calc in calculations:
            result = await self._verify_calculation(calc)
            if not result["correct"]:
                issues.append(f"Calculation error: {calc}")
                suggestions.append(f"Correct calculation: {result['corrected']}")
        
        # Check number consistency
        for number in numbers:
            consistency = await self._check_number_consistency(number, context)
            if consistency < 0.8:
                issues.append(f"Number may be inconsistent: {number}")
                suggestions.append(f"Verify number: {number}")
        
        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            confidence=0.9 if len(issues) == 0 else 0.4
        )
    
    def _extract_numbers(self, content: str) -> List[Dict[str, Any]]:
        """Extract numbers from content."""
        import re
        
        # Find numbers with context
        number_pattern = r'(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
        matches = re.finditer(number_pattern, content)
        
        numbers = []
        for match in matches:
            number = float(match.group(1))
            start, end = match.span()
            context_before = content[max(0, start-20):start]
            context_after = content[end:end+20]
            
            numbers.append({
                "value": number,
                "context": context_before + match.group(1) + context_after,
                "position": (start, end)
            })
        
        return numbers
    
    def _extract_calculations(self, content: str) -> List[str]:
        """Extract mathematical calculations from content."""
        import re
        
        # Look for calculation patterns
        calc_patterns = [
            r'(\d+(?:\.\d+)?)\s*[+\-*/]\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*%\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
            r'sum\s*of\s*(\d+(?:\.\d+)?)\s*and\s*(\d+(?:\.\d+)?)\s*is\s*(\d+(?:\.\d+)?)'
        ]
        
        calculations = []
        for pattern in calc_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                calculations.append(f"{match[0]} + {match[1]} = {match[2]}")
        
        return calculations
    
    async def _verify_calculation(self, calculation: str) -> Dict[str, Any]:
        """Verify a mathematical calculation."""
        try:
            # Parse and evaluate calculation
            parts = calculation.split('=')
            if len(parts) != 2:
                return {"correct": False, "corrected": calculation}
            
            expression = parts[0].strip()
            expected = float(parts[1].strip())
            
            # Safe evaluation (basic operations only)
            if re.match(r'^[\d\s+\-*/.()]+$', expression):
                actual = eval(expression)
                if abs(actual - expected) < self.tolerance:
                    return {"correct": True, "corrected": calculation}
                else:
                    return {"correct": False, "corrected": f"{expression} = {actual}"}
            
        except Exception:
            pass
        
        return {"correct": False, "corrected": calculation}
    
    async def _check_number_consistency(
        self,
        number: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Check if number is consistent with context."""
        # Simple heuristic: check if number is within reasonable bounds
        value = number["value"]
        
        if value < 0:
            return 0.3  # Negative numbers often need verification
        
        if value > 1000000:
            return 0.7  # Large numbers need verification
        
        return 0.9  # Default high confidence


class CodeChecker:
    """Code validation and testing."""
    
    def __init__(self):
        self.allowed_languages = ["python", "javascript", "sql"]
    
    async def check_code(
        self,
        code: str,
        language: str = "python"
    ) -> ValidationResult:
        """Check code for syntax and run tests."""
        
        if language not in self.allowed_languages:
            return ValidationResult(
                passed=False,
                issues=[f"Unsupported language: {language}"],
                suggestions=[f"Use one of: {', '.join(self.allowed_languages)}"]
            )
        
        issues = []
        suggestions = []
        
        # Syntax check
        syntax_result = await self._check_syntax(code, language)
        if not syntax_result["valid"]:
            issues.append(f"Syntax error: {syntax_result['error']}")
            suggestions.append(f"Fix syntax: {syntax_result['suggestion']}")
        
        # Run tests if available
        test_result = await self._run_tests(code, language)
        if not test_result["passed"]:
            issues.append(f"Tests failed: {test_result['error']}")
            suggestions.append(f"Fix tests: {test_result['suggestion']}")
        
        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            confidence=0.95 if len(issues) == 0 else 0.2
        )
    
    async def _check_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """Check code syntax."""
        try:
            if language == "python":
                compile(code, '<string>', 'exec')
                return {"valid": True, "error": None, "suggestion": None}
            elif language == "javascript":
                # Simple JS syntax check
                if code.count('{') == code.count('}'):
                    return {"valid": True, "error": None, "suggestion": None}
                else:
                    return {
                        "valid": False,
                        "error": "Mismatched braces",
                        "suggestion": "Check brace matching"
                    }
            elif language == "sql":
                # Basic SQL syntax check
                if any(keyword in code.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                    return {"valid": True, "error": None, "suggestion": None}
                else:
                    return {
                        "valid": False,
                        "error": "No valid SQL statement",
                        "suggestion": "Add SELECT, INSERT, UPDATE, or DELETE statement"
                    }
        
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "suggestion": f"Fix syntax error: {e.msg}"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "suggestion": "Check code syntax"
            }
        
        return {"valid": True, "error": None, "suggestion": None}
    
    async def _run_tests(self, code: str, language: str) -> Dict[str, Any]:
        """Run tests on code."""
        try:
            if language == "python" and "def test_" in code:
                # Run Python tests
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    f.flush()
                    
                    result = subprocess.run(
                        ['python', '-m', 'pytest', f.name, '-v'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        return {"passed": True, "error": None, "suggestion": None}
                    else:
                        return {
                            "passed": False,
                            "error": result.stderr,
                            "suggestion": "Fix failing tests"
                        }
        
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "error": "Test timeout",
                "suggestion": "Optimize test performance"
            }
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "suggestion": "Check test implementation"
            }
        
        return {"passed": True, "error": None, "suggestion": None}


class PolicyChecker:
    """Security and policy validation."""
    
    def __init__(self):
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
    
    async def check_policies(
        self,
        content: str,
        policies: List[str]
    ) -> ValidationResult:
        """Check content against security policies."""
        
        issues = []
        suggestions = []
        
        # Check PII
        pii_result = await self._check_pii(content)
        if not pii_result["passed"]:
            issues.extend(pii_result["issues"])
            suggestions.extend(pii_result["suggestions"])
        
        # Check network access
        network_result = await self._check_network_access(content)
        if not network_result["passed"]:
            issues.extend(network_result["issues"])
            suggestions.extend(network_result["suggestions"])
        
        # Check file system access
        fs_result = await self._check_filesystem_access(content)
        if not fs_result["passed"]:
            issues.extend(fs_result["issues"])
            suggestions.extend(fs_result["suggestions"])
        
        return ValidationResult(
            passed=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            confidence=0.9 if len(issues) == 0 else 0.1
        )
    
    async def _check_pii(self, content: str) -> Dict[str, Any]:
        """Check for personally identifiable information."""
        issues = []
        suggestions = []
        
        for pattern in self.pii_patterns:
            matches = re.findall(pattern, content)
            if matches:
                issues.append(f"PII detected: {matches[0]}")
                suggestions.append("Remove or mask PII data")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions
        }
    
    async def _check_network_access(self, content: str) -> Dict[str, Any]:
        """Check for unauthorized network access."""
        issues = []
        suggestions = []
        
        # Check for network operations
        network_patterns = [
            r'requests\.get|requests\.post',
            r'urllib\.request',
            r'socket\.',
            r'http://|https://'
        ]
        
        for pattern in network_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Network access detected: {pattern}")
                suggestions.append("Ensure network access is authorized")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions
        }
    
    async def _check_filesystem_access(self, content: str) -> Dict[str, Any]:
        """Check for unauthorized file system access."""
        issues = []
        suggestions = []
        
        # Check for file operations
        fs_patterns = [
            r'open\([^)]*[\'"]w[\'"]',  # Write access
            r'os\.remove|os\.unlink',
            r'shutil\.rmtree',
            r'rm\s+',
            r'del\s+'
        ]
        
        for pattern in fs_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"File system write access detected: {pattern}")
                suggestions.append("Ensure file system access is authorized")
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions
        }


class Critic:
    """Main critic service for comprehensive validation."""
    
    def __init__(self):
        self.fact_checker = FactChecker()
        self.number_checker = NumberChecker()
        self.code_checker = CodeChecker()
        self.policy_checker = PolicyChecker()
    
    async def review_task(
        self,
        task_id: str,
        content: str,
        context: Dict[str, Any],
        project_id: str
    ) -> Dict[str, Any]:
        """Comprehensive task review."""
        
        review_id = str(uuid4())
        results = {
            "review_id": review_id,
            "task_id": task_id,
            "overall_passed": True,
            "checks": {},
            "issues": [],
            "suggestions": [],
            "confidence": 1.0
        }
        
        # Fact checking
        fact_result = await self.fact_checker.check_facts(content, project_id)
        results["checks"]["facts"] = {
            "passed": fact_result.passed,
            "issues": fact_result.issues,
            "suggestions": fact_result.suggestions,
            "confidence": fact_result.confidence
        }
        
        # Number checking
        number_result = await self.number_checker.check_numbers(content, context)
        results["checks"]["numbers"] = {
            "passed": number_result.passed,
            "issues": number_result.issues,
            "suggestions": number_result.suggestions,
            "confidence": number_result.confidence
        }
        
        # Code checking (if applicable)
        if self._contains_code(content):
            code_result = await self.code_checker.check_code(content)
            results["checks"]["code"] = {
                "passed": code_result.passed,
                "issues": code_result.issues,
                "suggestions": code_result.suggestions,
                "confidence": code_result.confidence
            }
        
        # Policy checking
        policy_result = await self.policy_checker.check_policies(content, [])
        results["checks"]["policies"] = {
            "passed": policy_result.passed,
            "issues": policy_result.issues,
            "suggestions": policy_result.suggestions,
            "confidence": policy_result.confidence
        }
        
        # Aggregate results
        all_passed = all(
            check["passed"] for check in results["checks"].values()
        )
        
        results["overall_passed"] = all_passed
        results["confidence"] = min(
            check["confidence"] for check in results["checks"].values()
        )
        
        # Collect all issues and suggestions
        for check in results["checks"].values():
            results["issues"].extend(check["issues"])
            results["suggestions"].extend(check["suggestions"])
        
        return results
    
    def _contains_code(self, content: str) -> bool:
        """Check if content contains code."""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ',
            'function ', 'var ', 'let ', 'const ',
            'SELECT ', 'INSERT ', 'UPDATE ', 'DELETE '
        ]
        
        return any(indicator in content for indicator in code_indicators)
    
    async def health_check(self) -> bool:
        """Check critic health."""
        return True


# Global critic instance
critic = Critic()
