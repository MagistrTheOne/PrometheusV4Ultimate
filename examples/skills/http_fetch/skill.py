"""HTTP Fetch skill implementation."""

import json
import os
import urllib.parse
from typing import Dict, Any, List, Optional, Union
from libs.skills import BaseSkill, SkillSpec, SkillRunResult, PermissionType, ResourceLimit


class HTTPFetchSkill(BaseSkill):
    """Skill for fetching data from HTTP endpoints with security restrictions."""
    
    def __init__(self):
        spec = SkillSpec(
            name="http_fetch",
            version="1.0.0",
            description="Fetch data from HTTP endpoints with allowlist domain restrictions",
            inputs={
                "url": "HTTP URL to fetch data from",
                "method": "HTTP method: GET, POST (default: GET)",
                "headers": "JSON string of HTTP headers (optional)",
                "data": "Request body data for POST requests (optional)",
                "output_file": "Path to save response data",
                "format": "Response format: json, text, binary (default: auto-detect)"
            },
            outputs={
                "output_file": "Path to the saved response file",
                "status_code": "HTTP status code",
                "content_type": "Response content type",
                "size_bytes": "Size of response in bytes",
                "headers": "Response headers as JSON"
            },
            perms={
                PermissionType.FS_READ: True,
                PermissionType.FS_WRITE: True,
                PermissionType.NETWORK: True,
                PermissionType.ENV_VAR: False
            },
            limits={
                ResourceLimit.CPU_MS: 10000,
                ResourceLimit.RAM_MB: 100,
                ResourceLimit.TIME_S: 60,
                ResourceLimit.DISK_MB: 100
            },
            tags=["http", "fetch", "api", "network", "data"],
            author="PrometheusULTIMATE",
            license="MIT"
        )
        super().__init__(spec)
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute HTTP fetch operation."""
        url = kwargs["url"]
        method = kwargs.get("method", "GET").upper()
        headers_str = kwargs.get("headers", "{}")
        data = kwargs.get("data", "")
        output_file = kwargs["output_file"]
        format_type = kwargs.get("format", "auto")
        
        # Validate URL
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        # Check domain allowlist (in production, this would be configurable)
        if not self._is_allowed_domain(url):
            raise ValueError(f"Domain not in allowlist: {url}")
        
        # Parse headers
        try:
            headers = json.loads(headers_str) if headers_str else {}
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in headers parameter")
        
        # Set default headers
        if method == "GET":
            headers.setdefault("User-Agent", "PrometheusULTIMATE-HTTPFetch/1.0")
        elif method == "POST":
            headers.setdefault("User-Agent", "PrometheusULTIMATE-HTTPFetch/1.0")
            headers.setdefault("Content-Type", "application/json")
        
        # Make HTTP request
        response_data = self._make_request(url, method, headers, data)
        
        # Save response
        self._save_response(output_file, response_data, format_type)
        
        return {
            "output_file": output_file,
            "status_code": response_data["status_code"],
            "content_type": response_data["content_type"],
            "size_bytes": response_data["size_bytes"],
            "headers": json.dumps(response_data["headers"])
        }
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _is_allowed_domain(self, url: str) -> bool:
        """Check if domain is in allowlist."""
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            
            # Allowlist of safe domains (in production, this would be configurable)
            allowed_domains = [
                "httpbin.org",
                "jsonplaceholder.typicode.com",
                "api.github.com",
                "api.openweathermap.org",
                "localhost",
                "127.0.0.1"
            ]
            
            # Check exact match or subdomain
            for allowed in allowed_domains:
                if domain == allowed or domain.endswith(f".{allowed}"):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _make_request(
        self, 
        url: str, 
        method: str, 
        headers: Dict[str, str], 
        data: str
    ) -> Dict[str, Any]:
        """Make HTTP request and return response data."""
        
        # In a real implementation, this would use httpx or requests
        # For now, we'll simulate the response based on the URL
        
        if "httpbin.org" in url:
            return self._simulate_httpbin_response(url, method, headers, data)
        elif "jsonplaceholder.typicode.com" in url:
            return self._simulate_jsonplaceholder_response(url, method, headers, data)
        else:
            return self._simulate_generic_response(url, method, headers, data)
    
    def _simulate_httpbin_response(
        self, 
        url: str, 
        method: str, 
        headers: Dict[str, str], 
        data: str
    ) -> Dict[str, Any]:
        """Simulate httpbin.org response."""
        
        if "/get" in url:
            response_data = {
                "url": url,
                "method": method,
                "headers": headers,
                "args": {}
            }
        elif "/post" in url:
            response_data = {
                "url": url,
                "method": method,
                "headers": headers,
                "data": data,
                "json": json.loads(data) if data and data.strip() else None
            }
        else:
            response_data = {
                "message": "Simulated response",
                "url": url,
                "method": method
            }
        
        return {
            "status_code": 200,
            "content_type": "application/json",
            "size_bytes": len(json.dumps(response_data)),
            "headers": {
                "Content-Type": "application/json",
                "Server": "httpbin.org"
            },
            "body": json.dumps(response_data, indent=2)
        }
    
    def _simulate_jsonplaceholder_response(
        self, 
        url: str, 
        method: str, 
        headers: Dict[str, str], 
        data: str
    ) -> Dict[str, Any]:
        """Simulate jsonplaceholder.typicode.com response."""
        
        if "/posts" in url:
            response_data = [
                {
                    "userId": 1,
                    "id": 1,
                    "title": "Sample Post Title",
                    "body": "This is a sample post body content."
                }
            ]
        elif "/users" in url:
            response_data = [
                {
                    "id": 1,
                    "name": "Sample User",
                    "username": "sampleuser",
                    "email": "sample@example.com"
                }
            ]
        else:
            response_data = {"message": "Simulated response from jsonplaceholder"}
        
        return {
            "status_code": 200,
            "content_type": "application/json",
            "size_bytes": len(json.dumps(response_data)),
            "headers": {
                "Content-Type": "application/json",
                "Server": "jsonplaceholder.typicode.com"
            },
            "body": json.dumps(response_data, indent=2)
        }
    
    def _simulate_generic_response(
        self, 
        url: str, 
        method: str, 
        headers: Dict[str, str], 
        data: str
    ) -> Dict[str, Any]:
        """Simulate generic HTTP response."""
        
        response_data = {
            "message": "Simulated HTTP response",
            "url": url,
            "method": method,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        return {
            "status_code": 200,
            "content_type": "application/json",
            "size_bytes": len(json.dumps(response_data)),
            "headers": {
                "Content-Type": "application/json",
                "Server": "PrometheusULTIMATE-Simulator"
            },
            "body": json.dumps(response_data, indent=2)
        }
    
    def _save_response(self, output_file: str, response_data: Dict[str, Any], format_type: str) -> None:
        """Save response data to file."""
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Determine format
        if format_type == "auto":
            content_type = response_data.get("content_type", "")
            if "json" in content_type:
                format_type = "json"
            elif "text" in content_type:
                format_type = "text"
            else:
                format_type = "text"
        
        # Save based on format
        if format_type == "json":
            # Pretty print JSON
            try:
                parsed_json = json.loads(response_data["body"])
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(parsed_json, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                # If not valid JSON, save as text
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(response_data["body"])
        else:
            # Save as text
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response_data["body"])
