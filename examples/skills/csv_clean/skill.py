"""CSV Clean skill implementation."""

import csv
import os
import re
from typing import Dict, Any, List, Optional, Union
from libs.skills import BaseSkill, SkillSpec, SkillRunResult, PermissionType, ResourceLimit


class CSVCleanSkill(BaseSkill):
    """Skill for cleaning and normalizing CSV data."""
    
    def __init__(self):
        spec = SkillSpec(
            name="csv_clean",
            version="1.0.0",
            description="Clean and normalize CSV data: handle missing values, type conversion, duplicates",
            inputs={
                "input_file": "Path to input CSV file",
                "output_file": "Path for cleaned CSV file",
                "missing_strategy": "Strategy for missing values: drop, fill, keep (default: keep)",
                "fill_value": "Value to fill missing data (default: empty string)",
                "remove_duplicates": "Remove duplicate rows (default: false)",
                "type_conversion": "Convert data types: auto, none (default: auto)"
            },
            outputs={
                "output_file": "Path to the cleaned CSV file",
                "rows_before": "Number of rows before cleaning",
                "rows_after": "Number of rows after cleaning",
                "columns_cleaned": "List of columns that were processed",
                "duplicates_removed": "Number of duplicate rows removed"
            },
            perms={
                PermissionType.FS_READ: True,
                PermissionType.FS_WRITE: True,
                PermissionType.NETWORK: False,
                PermissionType.ENV_VAR: False
            },
            limits={
                ResourceLimit.CPU_MS: 3000,
                ResourceLimit.RAM_MB: 150,
                ResourceLimit.TIME_S: 20,
                ResourceLimit.DISK_MB: 50
            },
            tags=["csv", "data", "clean", "normalize", "preprocessing"],
            author="PrometheusULTIMATE",
            license="MIT"
        )
        super().__init__(spec)
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute CSV cleaning operation."""
        input_file = kwargs["input_file"]
        output_file = kwargs["output_file"]
        missing_strategy = kwargs.get("missing_strategy", "keep")
        fill_value = kwargs.get("fill_value", "")
        remove_duplicates = kwargs.get("remove_duplicates", False)
        type_conversion = kwargs.get("type_conversion", "auto")
        
        # Validate input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Read input CSV
        data = self._read_csv(input_file)
        if not data:
            raise ValueError(f"Input CSV file is empty: {input_file}")
        
        rows_before = len(data)
        columns_cleaned = []
        duplicates_removed = 0
        
        # Clean data
        cleaned_data = self._clean_data(
            data, 
            missing_strategy, 
            fill_value, 
            remove_duplicates,
            type_conversion,
            columns_cleaned
        )
        
        # Count duplicates removed
        if remove_duplicates:
            duplicates_removed = rows_before - len(cleaned_data)
        
        # Write cleaned CSV
        self._write_csv(output_file, cleaned_data)
        
        return {
            "output_file": output_file,
            "rows_before": rows_before,
            "rows_after": len(cleaned_data),
            "columns_cleaned": columns_cleaned,
            "duplicates_removed": duplicates_removed
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
    
    def _clean_data(
        self, 
        data: List[Dict[str, str]], 
        missing_strategy: str,
        fill_value: str,
        remove_duplicates: bool,
        type_conversion: str,
        columns_cleaned: List[str]
    ) -> List[Dict[str, str]]:
        """Clean the data according to specified strategies."""
        
        if not data:
            return data
        
        cleaned_data = []
        seen_rows = set() if remove_duplicates else None
        
        for row in data:
            # Handle missing values
            cleaned_row = self._handle_missing_values(row, missing_strategy, fill_value)
            
            # Type conversion
            if type_conversion == "auto":
                cleaned_row = self._convert_types(cleaned_row, columns_cleaned)
            
            # Remove duplicates
            if remove_duplicates:
                row_key = tuple(sorted(cleaned_row.items()))
                if row_key in seen_rows:
                    continue  # Skip duplicate
                seen_rows.add(row_key)
            
            cleaned_data.append(cleaned_row)
        
        return cleaned_data
    
    def _handle_missing_values(
        self, 
        row: Dict[str, str], 
        strategy: str, 
        fill_value: str
    ) -> Dict[str, str]:
        """Handle missing values in a row."""
        cleaned_row = {}
        
        for key, value in row.items():
            if self._is_missing_value(value):
                if strategy == "drop":
                    # Skip this row entirely if any value is missing
                    return None
                elif strategy == "fill":
                    cleaned_row[key] = fill_value
                else:  # keep
                    cleaned_row[key] = value
            else:
                cleaned_row[key] = value
        
        return cleaned_row
    
    def _is_missing_value(self, value: str) -> bool:
        """Check if a value is considered missing."""
        if value is None:
            return True
        
        value_str = str(value).strip()
        missing_indicators = ["", "null", "none", "nan", "n/a", "na", "-", "?"]
        
        return value_str.lower() in missing_indicators
    
    def _convert_types(self, row: Dict[str, str], columns_cleaned: List[str]) -> Dict[str, str]:
        """Convert data types automatically."""
        converted_row = {}
        
        for key, value in row.items():
            if self._is_missing_value(value):
                converted_row[key] = value
                continue
            
            # Try to convert to appropriate type
            converted_value = self._try_convert_value(value)
            if converted_value != value:
                if key not in columns_cleaned:
                    columns_cleaned.append(key)
            
            converted_row[key] = converted_value
        
        return converted_row
    
    def _try_convert_value(self, value: str) -> Union[str, int, float, bool]:
        """Try to convert a string value to appropriate type."""
        value = value.strip()
        
        # Try boolean
        if value.lower() in ["true", "false", "yes", "no", "1", "0"]:
            if value.lower() in ["true", "yes", "1"]:
                return "true"
            elif value.lower() in ["false", "no", "0"]:
                return "false"
        
        # Try integer
        try:
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return str(int(value))
        except ValueError:
            pass
        
        # Try float
        try:
            float_val = float(value)
            # Only convert if it's not a whole number (to avoid converting "1" to "1.0")
            if float_val != int(float_val):
                return str(float_val)
            else:
                return str(int(float_val))
        except ValueError:
            pass
        
        # Try date (basic format detection)
        if self._looks_like_date(value):
            return self._normalize_date(value)
        
        # Return as string
        return value
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if value looks like a date."""
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return True
        
        return False
    
    def _normalize_date(self, value: str) -> str:
        """Normalize date to ISO format."""
        # Simple normalization - in production, use proper date parsing
        if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            return value  # Already in ISO format
        
        # For other formats, return as-is for now
        # In production, would parse and convert to ISO format
        return value
