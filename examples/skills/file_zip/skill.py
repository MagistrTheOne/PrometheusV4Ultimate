"""File Zip skill implementation."""

import json
import os
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from libs.skills import BaseSkill, SkillSpec, SkillRunResult, PermissionType, ResourceLimit


class FileZipSkill(BaseSkill):
    """Skill for creating ZIP archives from files and directories."""
    
    def __init__(self):
        spec = SkillSpec(
            name="file_zip",
            version="1.0.0",
            description="Create ZIP archives from files and directories",
            inputs={
                "input_paths": "JSON array of file/directory paths to zip",
                "output_file": "Path for the output ZIP file",
                "compression_level": "Compression level 0-9 (default: 6)",
                "include_hidden": "Include hidden files (default: false)"
            },
            outputs={
                "output_file": "Path to the created ZIP file",
                "files_count": "Number of files added to archive",
                "total_size": "Total size of files before compression",
                "compressed_size": "Size of the ZIP file",
                "compression_ratio": "Compression ratio (0-1)"
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
                ResourceLimit.TIME_S: 120,
                ResourceLimit.DISK_MB: 500
            },
            tags=["zip", "archive", "compression", "files"],
            author="PrometheusULTIMATE",
            license="MIT"
        )
        super().__init__(spec)
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute ZIP creation operation."""
        input_paths_str = kwargs["input_paths"]
        output_file = kwargs["output_file"]
        compression_level = int(kwargs.get("compression_level", 6))
        include_hidden = kwargs.get("include_hidden", False)
        
        # Parse input paths
        try:
            input_paths = json.loads(input_paths_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in input_paths parameter")
        
        if not isinstance(input_paths, list):
            raise ValueError("input_paths must be a JSON array")
        
        # Validate input paths
        for path in input_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input path does not exist: {path}")
        
        # Validate compression level
        if not 0 <= compression_level <= 9:
            raise ValueError("Compression level must be between 0 and 9")
        
        # Create ZIP file
        files_count, total_size = self._create_zip(
            input_paths, 
            output_file, 
            compression_level, 
            include_hidden
        )
        
        # Get compressed size
        compressed_size = os.path.getsize(output_file)
        
        # Calculate compression ratio
        compression_ratio = 1 - (compressed_size / total_size) if total_size > 0 else 0
        
        return {
            "output_file": output_file,
            "files_count": files_count,
            "total_size": total_size,
            "compressed_size": compressed_size,
            "compression_ratio": round(compression_ratio, 3)
        }
    
    def _create_zip(
        self, 
        input_paths: List[str], 
        output_file: str, 
        compression_level: int,
        include_hidden: bool
    ) -> tuple[int, int]:
        """Create ZIP file from input paths."""
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        files_count = 0
        total_size = 0
        
        with zipfile.ZipFile(
            output_file, 
            'w', 
            zipfile.ZIP_DEFLATED, 
            compresslevel=compression_level
        ) as zipf:
            
            for input_path in input_paths:
                path = Path(input_path)
                
                if path.is_file():
                    # Add single file
                    if self._should_include_file(path, include_hidden):
                        zipf.write(path, path.name)
                        files_count += 1
                        total_size += path.stat().st_size
                
                elif path.is_dir():
                    # Add directory recursively
                    for file_path in path.rglob('*'):
                        if file_path.is_file() and self._should_include_file(file_path, include_hidden):
                            # Calculate relative path within the archive
                            arcname = file_path.relative_to(path.parent)
                            zipf.write(file_path, arcname)
                            files_count += 1
                            total_size += file_path.stat().st_size
        
        return files_count, total_size
    
    def _should_include_file(self, file_path: Path, include_hidden: bool) -> bool:
        """Check if file should be included in the archive."""
        
        # Check if file is hidden
        if not include_hidden:
            # Check if any part of the path is hidden (starts with .)
            for part in file_path.parts:
                if part.startswith('.'):
                    return False
        
        # Check file size (avoid extremely large files)
        try:
            file_size = file_path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                return False
        except OSError:
            return False
        
        return True
