"""Plot Basic skill implementation."""

import csv
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from libs.skills import BaseSkill, SkillSpec, SkillRunResult, PermissionType, ResourceLimit


class PlotBasicSkill(BaseSkill):
    """Skill for creating basic plots and charts from data."""
    
    def __init__(self):
        spec = SkillSpec(
            name="plot_basic",
            version="1.0.0",
            description="Create basic plots and charts from data",
            inputs={
                "data_file": "Path to CSV file with data to plot",
                "output_file": "Path for output plot image",
                "plot_type": "Type of plot: line, bar, scatter, histogram (default: line)",
                "x_column": "Name of X-axis column",
                "y_column": "Name of Y-axis column",
                "title": "Plot title (optional)",
                "width": "Plot width in pixels (default: 800)",
                "height": "Plot height in pixels (default: 600)"
            },
            outputs={
                "output_file": "Path to the created plot image",
                "plot_type": "Type of plot created",
                "data_points": "Number of data points plotted",
                "image_size": "Size of output image in bytes",
                "dimensions": "Plot dimensions (width x height)"
            },
            perms={
                PermissionType.FS_READ: True,
                PermissionType.FS_WRITE: True,
                PermissionType.NETWORK: False,
                PermissionType.ENV_VAR: False
            },
            limits={
                ResourceLimit.CPU_MS: 8000,
                ResourceLimit.RAM_MB: 150,
                ResourceLimit.TIME_S: 30,
                ResourceLimit.DISK_MB: 50
            },
            tags=["plot", "chart", "visualization", "data", "graph"],
            author="PrometheusULTIMATE",
            license="MIT"
        )
        super().__init__(spec)
    
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute plot creation."""
        data_file = kwargs["data_file"]
        output_file = kwargs["output_file"]
        plot_type = kwargs.get("plot_type", "line")
        x_column = kwargs["x_column"]
        y_column = kwargs["y_column"]
        title = kwargs.get("title", f"{plot_type.title()} Plot")
        width = int(kwargs.get("width", 800))
        height = int(kwargs.get("height", 600))
        
        # Validate input file
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Validate plot type
        if plot_type not in ["line", "bar", "scatter", "histogram"]:
            raise ValueError(f"Invalid plot type: {plot_type}")
        
        # Read data
        data = self._read_csv_data(data_file)
        
        # Validate columns
        if x_column not in data["headers"]:
            raise ValueError(f"X column '{x_column}' not found. Available: {data['headers']}")
        if y_column not in data["headers"]:
            raise ValueError(f"Y column '{y_column}' not found. Available: {data['headers']}")
        
        # Create plot
        plot_data = self._create_plot(
            data, x_column, y_column, plot_type, title, width, height
        )
        
        # Save plot
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(plot_data, f, indent=2)
        
        # Get file size
        image_size = os.path.getsize(output_file)
        
        return {
            "output_file": output_file,
            "plot_type": plot_type,
            "data_points": len(data["rows"]),
            "image_size": image_size,
            "dimensions": f"{width}x{height}"
        }
    
    def _read_csv_data(self, file_path: str) -> Dict[str, Any]:
        """Read CSV data and return structured data."""
        data = []
        headers = []
        
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames
            
            for row in reader:
                data.append(row)
        
        return {
            "headers": headers,
            "rows": data
        }
    
    def _create_plot(
        self, 
        data: Dict[str, Any], 
        x_column: str, 
        y_column: str, 
        plot_type: str, 
        title: str, 
        width: int, 
        height: int
    ) -> Dict[str, Any]:
        """Create plot data structure."""
        
        # Extract data for plotting
        x_data = []
        y_data = []
        
        for row in data["rows"]:
            try:
                x_val = float(row[x_column]) if row[x_column] else 0
                y_val = float(row[y_column]) if row[y_column] else 0
                x_data.append(x_val)
                y_data.append(y_val)
            except (ValueError, TypeError):
                # Skip invalid data points
                continue
        
        # Create plot configuration
        plot_config = {
            "type": plot_type,
            "title": title,
            "width": width,
            "height": height,
            "x_axis": {
                "label": x_column,
                "data": x_data
            },
            "y_axis": {
                "label": y_column,
                "data": y_data
            },
            "data_points": len(x_data),
            "metadata": {
                "created_by": "PrometheusULTIMATE Plot Basic Skill",
                "version": "1.0.0",
                "data_file": data.get("source_file", "unknown")
            }
        }
        
        # Add plot-specific configurations
        if plot_type == "line":
            plot_config["line_style"] = {
                "color": "#007bff",
                "width": 2,
                "marker": "circle"
            }
        elif plot_type == "bar":
            plot_config["bar_style"] = {
                "color": "#28a745",
                "width": 0.8
            }
        elif plot_type == "scatter":
            plot_config["scatter_style"] = {
                "color": "#dc3545",
                "size": 50,
                "alpha": 0.7
            }
        elif plot_type == "histogram":
            plot_config["histogram_style"] = {
                "color": "#ffc107",
                "bins": 20,
                "alpha": 0.7
            }
        
        return plot_config
