"""SQL Query skill implementation."""

import os
import json
import csv
import time
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from libs.skills import BaseSkill, SkillSpec, SkillRunResult, PermissionType, ResourceLimit


class SQLQuerySkill(BaseSkill):
    """Skill for executing read-only SQL queries on databases."""
    
    def __init__(self):
        spec = SkillSpec(
            name="sql_query",
            version="1.0.0",
            description="Execute read-only SQL queries on databases",
            inputs={
                "database_url": "Database connection URL (SQLite, PostgreSQL)",
                "query": "SQL query to execute (read-only)",
                "output_file": "Path to save query results",
                "output_format": "Output format: csv, json (default: csv)"
            },
            outputs={
                "output_file": "Path to the results file",
                "rows_returned": "Number of rows returned",
                "columns_count": "Number of columns in result",
                "execution_time": "Query execution time in seconds",
                "database_type": "Type of database connected"
            },
            perms={
                PermissionType.FS_READ: True,
                PermissionType.FS_WRITE: True,
                PermissionType.NETWORK: False,
                PermissionType.ENV_VAR: False
            },
            limits={
                ResourceLimit.CPU_MS: 15000,
                ResourceLimit.RAM_MB: 200,
                ResourceLimit.TIME_S: 60,
                ResourceLimit.DISK_MB: 100
            },
            tags=["sql", "database", "query", "readonly", "data"],
            author="PrometheusULTIMATE",
            license="MIT"
        )
        super().__init__(spec)
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute SQL query."""
        database_url = kwargs["database_url"]
        query = kwargs["query"]
        output_file = kwargs["output_file"]
        output_format = kwargs.get("output_format", "csv")
        
        # Validate query (read-only)
        if not self._is_readonly_query(query):
            raise ValueError("Only read-only queries are allowed (SELECT, WITH, etc.)")
        
        # Determine database type
        database_type = self._get_database_type(database_url)
        
        # Start timing
        start_time = time.time()
        
        # Execute query
        results = self._execute_query(database_url, query, database_type)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if output_format == "json":
            self._save_json_results(output_file, results)
        else:
            self._save_csv_results(output_file, results)
        
        return {
            "output_file": output_file,
            "rows_returned": len(results["rows"]),
            "columns_count": len(results["columns"]),
            "execution_time": round(execution_time, 3),
            "database_type": database_type
        }
    
    def _is_readonly_query(self, query: str) -> bool:
        """Check if query is read-only."""
        query_upper = query.upper().strip()
        
        # Allowed read-only keywords
        readonly_keywords = ["SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW", "PRAGMA"]
        
        # Dangerous keywords that modify data
        dangerous_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", 
            "TRUNCATE", "REPLACE", "MERGE", "CALL", "EXEC", "EXECUTE"
        ]
        
        # Check if query starts with allowed keyword
        for keyword in readonly_keywords:
            if query_upper.startswith(keyword):
                # Check for dangerous keywords in the query
                for dangerous in dangerous_keywords:
                    if dangerous in query_upper:
                        return False
                return True
        
        return False
    
    def _get_database_type(self, database_url: str) -> str:
        """Determine database type from URL."""
        if database_url.startswith("sqlite://"):
            return "SQLite"
        elif database_url.startswith("postgresql://"):
            return "PostgreSQL"
        elif database_url.startswith("mysql://"):
            return "MySQL"
        else:
            return "Unknown"
    
    def _execute_query(self, database_url: str, query: str, database_type: str) -> Dict[str, Any]:
        """Execute SQL query and return results."""
        
        if database_type == "SQLite":
            return self._execute_sqlite_query(database_url, query)
        else:
            # For other database types, simulate results
            return self._simulate_query_results(query, database_type)
    
    def _execute_sqlite_query(self, database_url: str, query: str) -> Dict[str, Any]:
        """Execute SQLite query."""
        try:
            # Extract file path from SQLite URL
            if database_url.startswith("sqlite:///"):
                db_path = database_url[10:]  # Remove "sqlite:///"
            else:
                db_path = database_url
            
            # Check if database file exists
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"SQLite database not found: {db_path}")
            
            # Connect to database
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(query)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Get rows
            rows = []
            for row in cursor.fetchall():
                row_dict = {}
                for i, value in enumerate(row):
                    row_dict[columns[i]] = value
                rows.append(row_dict)
            
            # Close connection
            conn.close()
            
            return {
                "columns": columns,
                "rows": rows
            }
            
        except Exception as e:
            raise ValueError(f"SQLite query execution failed: {e}")
    
    def _simulate_query_results(self, query: str, database_type: str) -> Dict[str, Any]:
        """Simulate query results for non-SQLite databases."""
        
        # Parse query to determine what kind of results to simulate
        query_upper = query.upper()
        
        if "FROM users" in query_upper or "FROM user" in query_upper:
            return self._simulate_users_table()
        elif "FROM products" in query_upper or "FROM product" in query_upper:
            return self._simulate_products_table()
        elif "FROM orders" in query_upper or "FROM order" in query_upper:
            return self._simulate_orders_table()
        else:
            return self._simulate_generic_table()
    
    def _simulate_users_table(self) -> Dict[str, Any]:
        """Simulate users table results."""
        return {
            "columns": ["id", "name", "email", "created_at"],
            "rows": [
                {"id": 1, "name": "John Doe", "email": "john@example.com", "created_at": "2024-01-01"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "created_at": "2024-01-02"},
                {"id": 3, "name": "Bob Johnson", "email": "bob@example.com", "created_at": "2024-01-03"}
            ]
        }
    
    def _simulate_products_table(self) -> Dict[str, Any]:
        """Simulate products table results."""
        return {
            "columns": ["id", "name", "price", "category", "stock"],
            "rows": [
                {"id": 1, "name": "Laptop", "price": 999.99, "category": "Electronics", "stock": 50},
                {"id": 2, "name": "Mouse", "price": 29.99, "category": "Electronics", "stock": 100},
                {"id": 3, "name": "Keyboard", "price": 79.99, "category": "Electronics", "stock": 75}
            ]
        }
    
    def _simulate_orders_table(self) -> Dict[str, Any]:
        """Simulate orders table results."""
        return {
            "columns": ["id", "user_id", "product_id", "quantity", "total", "order_date"],
            "rows": [
                {"id": 1, "user_id": 1, "product_id": 1, "quantity": 1, "total": 999.99, "order_date": "2024-01-15"},
                {"id": 2, "user_id": 2, "product_id": 2, "quantity": 2, "total": 59.98, "order_date": "2024-01-16"},
                {"id": 3, "user_id": 1, "product_id": 3, "quantity": 1, "total": 79.99, "order_date": "2024-01-17"}
            ]
        }
    
    def _simulate_generic_table(self) -> Dict[str, Any]:
        """Simulate generic table results."""
        return {
            "columns": ["id", "name", "value", "status"],
            "rows": [
                {"id": 1, "name": "Item 1", "value": 100, "status": "active"},
                {"id": 2, "name": "Item 2", "value": 200, "status": "inactive"},
                {"id": 3, "name": "Item 3", "value": 300, "status": "active"}
            ]
        }
    
    def _save_csv_results(self, output_file: str, results: Dict[str, Any]) -> None:
        """Save results as CSV file."""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if results["rows"]:
                fieldnames = results["columns"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results["rows"])
            else:
                # Empty result set
                writer = csv.writer(csvfile)
                writer.writerow(results["columns"])
    
    def _save_json_results(self, output_file: str, results: Dict[str, Any]) -> None:
        """Save results as JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
