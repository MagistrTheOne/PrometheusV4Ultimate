"""CSV Join skill implementation."""

import csv
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from libs.skills import BaseSkill, SkillSpec, SkillRunResult, PermissionType, ResourceLimit


class CSVJoinSkill(BaseSkill):
    """Skill for joining two CSV files by a common key."""
    
    def __init__(self):
        spec = SkillSpec(
            name="csv_join",
            version="1.0.0",
            description="Join two CSV files by a common key column",
            inputs={
                "left_file": "Path to left CSV file",
                "right_file": "Path to right CSV file", 
                "left_key": "Key column name in left CSV",
                "right_key": "Key column name in right CSV",
                "join_type": "Type of join: inner, left, right, outer (default: inner)",
                "output_file": "Path for output CSV file"
            },
            outputs={
                "output_file": "Path to the created joined CSV file",
                "rows_count": "Number of rows in the output",
                "columns_count": "Number of columns in the output"
            },
            perms={
                PermissionType.FS_READ: True,
                PermissionType.FS_WRITE: True,
                PermissionType.NETWORK: False,
                PermissionType.ENV_VAR: False
            },
            limits={
                ResourceLimit.CPU_MS: 5000,
                ResourceLimit.RAM_MB: 200,
                ResourceLimit.TIME_S: 30,
                ResourceLimit.DISK_MB: 100
            },
            tags=["csv", "data", "join", "merge"],
            author="PrometheusULTIMATE",
            license="MIT"
        )
        super().__init__(spec)
    
    def run(self, **kwargs) -> SkillRunResult:
        """Execute the skill with given inputs."""
        try:
            # Validate inputs
            self._validate_inputs(kwargs)
            
            # Execute skill logic
            outputs = self._execute(**kwargs)
            
            return SkillRunResult(
                success=True,
                outputs=outputs,
                metrics={},
                logs=[]
            )
            
        except Exception as e:
            return SkillRunResult(
                success=False,
                outputs={},
                error=str(e),
                metrics={},
                logs=[]
            )
    
    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate input parameters against spec."""
        required_inputs = ["left_file", "right_file", "left_key", "right_key", "output_file"]
        for input_name in required_inputs:
            if input_name not in inputs:
                raise ValueError(f"Missing required input: {input_name}")
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute CSV join operation."""
        left_file = kwargs["left_file"]
        right_file = kwargs["right_file"]
        left_key = kwargs["left_key"]
        right_key = kwargs["right_key"]
        join_type = kwargs.get("join_type", "inner")
        output_file = kwargs["output_file"]
        
        # Validate input files exist
        if not os.path.exists(left_file):
            raise FileNotFoundError(f"Left file not found: {left_file}")
        if not os.path.exists(right_file):
            raise FileNotFoundError(f"Right file not found: {right_file}")
        
        # Read left CSV
        left_data = self._read_csv(left_file)
        if not left_data:
            raise ValueError(f"Left CSV file is empty: {left_file}")
        
        # Read right CSV
        right_data = self._read_csv(right_file)
        if not right_data:
            raise ValueError(f"Right CSV file is empty: {right_file}")
        
        # Validate key columns exist
        left_headers = left_data[0].keys()
        right_headers = right_data[0].keys()
        
        if left_key not in left_headers:
            raise ValueError(f"Key column '{left_key}' not found in left CSV. Available: {list(left_headers)}")
        if right_key not in right_headers:
            raise ValueError(f"Key column '{right_key}' not found in right CSV. Available: {list(right_headers)}")
        
        # Perform join
        joined_data = self._perform_join(left_data, right_data, left_key, right_key, join_type)
        
        # Write output
        self._write_csv(output_file, joined_data)
        
        return {
            "output_file": output_file,
            "rows_count": len(joined_data),
            "columns_count": len(joined_data[0]) if joined_data else 0
        }
    
    def _read_csv(self, file_path: str) -> List[Dict[str, str]]:
        """Read CSV file and return list of dictionaries."""
        data = []
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        return data
    
    def _write_csv(self, file_path: str, data: List[Dict[str, str]]) -> None:
        """Write data to CSV file."""
        if not data:
            # Create empty CSV with headers
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([])
            return
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    def _perform_join(
        self, 
        left_data: List[Dict[str, str]], 
        right_data: List[Dict[str, str]], 
        left_key: str, 
        right_key: str, 
        join_type: str
    ) -> List[Dict[str, str]]:
        """Perform the actual join operation."""
        
        # Create lookup dictionary for right data
        right_lookup = {}
        for row in right_data:
            key_value = row[right_key]
            if key_value not in right_lookup:
                right_lookup[key_value] = []
            right_lookup[key_value].append(row)
        
        result = []
        left_keys_used = set()
        
        # Process left data
        for left_row in left_data:
            left_key_value = left_row[left_key]
            left_keys_used.add(left_key_value)
            
            if left_key_value in right_lookup:
                # Match found - create joined rows
                for right_row in right_lookup[left_key_value]:
                    joined_row = self._merge_rows(left_row, right_row, left_key, right_key)
                    result.append(joined_row)
            elif join_type in ["left", "outer"]:
                # No match - include left row with nulls for right columns
                joined_row = self._merge_rows(left_row, {}, left_key, right_key)
                result.append(joined_row)
        
        # Handle right outer join
        if join_type in ["right", "outer"]:
            for right_row in right_data:
                right_key_value = right_row[right_key]
                if right_key_value not in left_keys_used:
                    # No match in left - include right row with nulls for left columns
                    joined_row = self._merge_rows({}, right_row, left_key, right_key)
                    result.append(joined_row)
        
        return result
    
    def _merge_rows(
        self, 
        left_row: Dict[str, str], 
        right_row: Dict[str, str], 
        left_key: str, 
        right_key: str
    ) -> Dict[str, str]:
        """Merge two rows, handling key column conflicts."""
        merged = left_row.copy()
        
        for key, value in right_row.items():
            if key == right_key:
                # Skip the right key column since we already have it from left
                continue
            elif key in merged and key != left_key:
                # Column name conflict - prefix with table name
                merged[f"right_{key}"] = value
            else:
                merged[key] = value
        
        return merged
