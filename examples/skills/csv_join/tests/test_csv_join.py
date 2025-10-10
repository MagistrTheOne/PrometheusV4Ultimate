"""Tests for CSV Join skill."""

import csv
import os
import tempfile
import pytest
from pathlib import Path
import sys

# Add parent directory to path to import skill
sys.path.insert(0, str(Path(__file__).parent.parent))

from skill import CSVJoinSkill


class TestCSVJoinSkill:
    """Test cases for CSV Join skill."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.skill = CSVJoinSkill()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV files
        self.left_file = os.path.join(self.temp_dir, "left.csv")
        self.right_file = os.path.join(self.temp_dir, "right.csv")
        self.output_file = os.path.join(self.temp_dir, "output.csv")
        
        self._create_test_files()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_files(self):
        """Create test CSV files."""
        # Left CSV: users
        left_data = [
            {"user_id": "1", "name": "Alice", "age": "25"},
            {"user_id": "2", "name": "Bob", "age": "30"},
            {"user_id": "3", "name": "Charlie", "age": "35"}
        ]
        
        with open(self.left_file, 'w', newline='') as csvfile:
            fieldnames = ["user_id", "name", "age"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(left_data)
        
        # Right CSV: orders
        right_data = [
            {"order_id": "101", "user_id": "1", "product": "Laptop", "amount": "1000"},
            {"order_id": "102", "user_id": "1", "product": "Mouse", "amount": "25"},
            {"order_id": "103", "user_id": "2", "product": "Keyboard", "amount": "75"},
            {"order_id": "104", "user_id": "4", "product": "Monitor", "amount": "300"}
        ]
        
        with open(self.right_file, 'w', newline='') as csvfile:
            fieldnames = ["order_id", "user_id", "product", "amount"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(right_data)
    
    def test_inner_join(self):
        """Test inner join operation."""
        result = self.skill.run(
            left_file=self.left_file,
            right_file=self.right_file,
            left_key="user_id",
            right_key="user_id",
            join_type="inner",
            output_file=self.output_file
        )
        
        assert result.success
        assert result.outputs["rows_count"] == 3  # Alice has 2 orders, Bob has 1
        assert result.outputs["columns_count"] == 6  # user_id, name, age, order_id, product, amount
        assert os.path.exists(self.output_file)
        
        # Verify output content
        with open(self.output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3
        # Check that Alice appears twice (2 orders)
        alice_orders = [row for row in rows if row["name"] == "Alice"]
        assert len(alice_orders) == 2
        
        # Check that Bob appears once (1 order)
        bob_orders = [row for row in rows if row["name"] == "Bob"]
        assert len(bob_orders) == 1
    
    def test_left_join(self):
        """Test left join operation."""
        result = self.skill.run(
            left_file=self.left_file,
            right_file=self.right_file,
            left_key="user_id",
            right_key="user_id",
            join_type="left",
            output_file=self.output_file
        )
        
        assert result.success
        assert result.outputs["rows_count"] == 4  # All users from left, Charlie has no orders
        assert result.outputs["columns_count"] == 6
        
        # Verify Charlie appears with empty order data
        with open(self.output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        charlie_rows = [row for row in rows if row["name"] == "Charlie"]
        assert len(charlie_rows) == 1
        assert charlie_rows[0]["order_id"] == ""  # No order data
    
    def test_right_join(self):
        """Test right join operation."""
        result = self.skill.run(
            left_file=self.left_file,
            right_file=self.right_file,
            left_key="user_id",
            right_key="user_id",
            join_type="right",
            output_file=self.output_file
        )
        
        assert result.success
        assert result.outputs["rows_count"] == 4  # All orders from right, user_id=4 has no user data
        assert result.outputs["columns_count"] == 6
        
        # Verify user_id=4 appears with empty user data
        with open(self.output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        user4_rows = [row for row in rows if row["user_id"] == "4"]
        assert len(user4_rows) == 1
        assert user4_rows[0]["name"] == ""  # No user data
    
    def test_outer_join(self):
        """Test outer join operation."""
        result = self.skill.run(
            left_file=self.left_file,
            right_file=self.right_file,
            left_key="user_id",
            right_key="user_id",
            join_type="outer",
            output_file=self.output_file
        )
        
        assert result.success
        assert result.outputs["rows_count"] == 5  # All users and all orders
        assert result.outputs["columns_count"] == 6
    
    def test_missing_file(self):
        """Test error handling for missing files."""
        result = self.skill.run(
            left_file="nonexistent.csv",
            right_file=self.right_file,
            left_key="user_id",
            right_key="user_id",
            output_file=self.output_file
        )
        
        assert not result.success
        assert "not found" in result.error
    
    def test_missing_key_column(self):
        """Test error handling for missing key columns."""
        result = self.skill.run(
            left_file=self.left_file,
            right_file=self.right_file,
            left_key="nonexistent_key",
            right_key="user_id",
            output_file=self.output_file
        )
        
        assert not result.success
        assert "not found" in result.error
    
    def test_empty_csv(self):
        """Test handling of empty CSV files."""
        # Create empty CSV
        empty_file = os.path.join(self.temp_dir, "empty.csv")
        with open(empty_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([])
        
        result = self.skill.run(
            left_file=empty_file,
            right_file=self.right_file,
            left_key="user_id",
            right_key="user_id",
            output_file=self.output_file
        )
        
        assert not result.success
        assert "empty" in result.error
    
    def test_column_name_conflict(self):
        """Test handling of column name conflicts."""
        # Create CSV with conflicting column names
        conflict_file = os.path.join(self.temp_dir, "conflict.csv")
        conflict_data = [
            {"user_id": "1", "name": "Alice", "age": "25", "product": "UserProduct"}
        ]
        
        with open(conflict_file, 'w', newline='') as csvfile:
            fieldnames = ["user_id", "name", "age", "product"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(conflict_data)
        
        result = self.skill.run(
            left_file=conflict_file,
            right_file=self.right_file,
            left_key="user_id",
            right_key="user_id",
            output_file=self.output_file
        )
        
        assert result.success
        # Should have right_product column to avoid conflict
        with open(self.output_file, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        
        assert "product" in headers  # From left
        assert "right_product" in headers  # From right (renamed)
    
    def test_spec_properties(self):
        """Test skill specification properties."""
        spec = self.skill.spec
        
        assert spec.name == "csv_join"
        assert spec.version == "1.0.0"
        assert "Join two CSV files" in spec.description
        assert "csv" in spec.tags
        assert "data" in spec.tags
        assert "join" in spec.tags
        assert "merge" in spec.tags
        assert spec.author == "PrometheusULTIMATE"
        assert spec.license == "MIT"
        
        # Check inputs
        assert "left_file" in spec.inputs
        assert "right_file" in spec.inputs
        assert "left_key" in spec.inputs
        assert "right_key" in spec.inputs
        assert "join_type" in spec.inputs
        assert "output_file" in spec.inputs
        
        # Check outputs
        assert "output_file" in spec.outputs
        assert "rows_count" in spec.outputs
        assert "columns_count" in spec.outputs
        
        # Check permissions
        assert spec.perms["fs_read"] is True
        assert spec.perms["fs_write"] is True
        assert spec.perms["network"] is False
        assert spec.perms["env_var"] is False
        
        # Check limits
        assert spec.limits["cpu_ms"] == 5000
        assert spec.limits["ram_mb"] == 200
        assert spec.limits["time_s"] == 30
        assert spec.limits["disk_mb"] == 100
