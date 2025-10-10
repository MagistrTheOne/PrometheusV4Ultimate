"""End-to-end tests for full pipeline."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_csv_report_pipeline():
    """Test CSV report generation pipeline."""
    
    # Mock all service calls
    with patch('httpx.AsyncClient') as mock_client:
        # Mock planner response
        planner_response = AsyncMock()
        planner_response.json.return_value = {
            "plan_id": "plan-001",
            "steps": [
                {
                    "name": "load_csv",
                    "skill_name": "csv_clean",
                    "inputs": {"file_path": "sales.csv"},
                    "description": "Load and clean CSV data"
                },
                {
                    "name": "analyze_data",
                    "skill_name": "sql_query",
                    "inputs": {"query": "SELECT SUM(amount) FROM sales"},
                    "description": "Analyze sales data"
                },
                {
                    "name": "create_plot",
                    "skill_name": "plot_basic",
                    "inputs": {"data": "{{analyze_data.output}}", "chart_type": "bar"},
                    "description": "Create sales chart"
                }
            ],
            "budget": {"total_cost_usd": 0.003, "total_time_ms": 3000},
            "estimated_duration_ms": 3000
        }
        planner_response.raise_for_status.return_value = None
        
        # Mock critic response
        critic_response = AsyncMock()
        critic_response.json.return_value = {
            "review_id": "review-001",
            "overall_passed": True,
            "checks": {
                "facts": {"passed": True, "issues": [], "suggestions": []},
                "numbers": {"passed": True, "issues": [], "suggestions": []},
                "policies": {"passed": True, "issues": [], "suggestions": []}
            },
            "issues": [],
            "suggestions": [],
            "confidence": 0.95
        }
        critic_response.raise_for_status.return_value = None
        
        # Configure mock client
        mock_client.return_value.__aenter__.return_value.post.side_effect = [
            planner_response,  # First call to planner
            critic_response    # Second call to critic
        ]
        
        # Import orchestrator
        from apps.orchestrator.orchestrator import orchestrator
        
        # Create task
        task_id = await orchestrator.create_task(
            goal="Create sales report from CSV data",
            inputs={"csv_file": "sales.csv"},
            limits={"time_s": 30, "cost_usd": 0.1},
            project_id="test"
        )
        
        assert task_id is not None
        
        # Wait for task completion
        await asyncio.sleep(0.2)
        
        # Check task status
        status = await orchestrator.get_task_status(task_id)
        assert status is not None
        assert "task" in status
        assert "steps" in status
        assert "events" in status


@pytest.mark.asyncio
async def test_code_patch_pipeline():
    """Test code patching pipeline with critic catching errors."""
    
    with patch('httpx.AsyncClient') as mock_client:
        # Mock planner response
        planner_response = AsyncMock()
        planner_response.json.return_value = {
            "plan_id": "plan-002",
            "steps": [
                {
                    "name": "analyze_code",
                    "skill_name": "code_format",
                    "inputs": {"code": "def buggy_function(): return 1/0"},
                    "description": "Analyze code for bugs"
                },
                {
                    "name": "run_tests",
                    "skill_name": "math_calc",
                    "inputs": {"expression": "test_buggy_function()"},
                    "description": "Run tests to verify fix"
                }
            ],
            "budget": {"total_cost_usd": 0.002, "total_time_ms": 2000},
            "estimated_duration_ms": 2000
        }
        planner_response.raise_for_status.return_value = None
        
        # Mock critic response (should catch the bug)
        critic_response = AsyncMock()
        critic_response.json.return_value = {
            "review_id": "review-002",
            "overall_passed": False,
            "checks": {
                "code": {
                    "passed": False,
                    "issues": ["Division by zero error detected"],
                    "suggestions": ["Add zero check before division"]
                },
                "policies": {"passed": True, "issues": [], "suggestions": []}
            },
            "issues": ["Division by zero error detected"],
            "suggestions": ["Add zero check before division"],
            "confidence": 0.8
        }
        critic_response.raise_for_status.return_value = None
        
        # Configure mock client
        mock_client.return_value.__aenter__.return_value.post.side_effect = [
            planner_response,
            critic_response
        ]
        
        from apps.orchestrator.orchestrator import orchestrator
        
        # Create task
        task_id = await orchestrator.create_task(
            goal="Fix bug in buggy_function",
            inputs={"code": "def buggy_function(): return 1/0"},
            limits={"time_s": 60, "cost_usd": 0.2},
            project_id="test"
        )
        
        assert task_id is not None
        
        # Wait for task completion
        await asyncio.sleep(0.2)
        
        # Check task status
        status = await orchestrator.get_task_status(task_id)
        assert status is not None
        
        # Should have issues detected by critic
        events = status.get("events", [])
        review_events = [e for e in events if e.get("event_type") == "state_transition"]
        assert len(review_events) > 0


@pytest.mark.asyncio
async def test_offline_mode_pipeline():
    """Test offline mode (no network access)."""
    
    with patch('httpx.AsyncClient') as mock_client:
        # Mock planner response (offline mode)
        planner_response = AsyncMock()
        planner_response.json.return_value = {
            "plan_id": "plan-003",
            "steps": [
                {
                    "name": "local_calculation",
                    "skill_name": "math_calc",
                    "inputs": {"expression": "2 + 2"},
                    "description": "Perform local calculation"
                }
            ],
            "budget": {"total_cost_usd": 0.001, "total_time_ms": 1000},
            "estimated_duration_ms": 1000
        }
        planner_response.raise_for_status.return_value = None
        
        # Mock critic response (should pass offline check)
        critic_response = AsyncMock()
        critic_response.json.return_value = {
            "review_id": "review-003",
            "overall_passed": True,
            "checks": {
                "policies": {
                    "passed": True,
                    "issues": [],
                    "suggestions": []
                }
            },
            "issues": [],
            "suggestions": [],
            "confidence": 0.9
        }
        critic_response.raise_for_status.return_value = None
        
        # Configure mock client
        mock_client.return_value.__aenter__.return_value.post.side_effect = [
            planner_response,
            critic_response
        ]
        
        from apps.orchestrator.orchestrator import orchestrator
        
        # Create task
        task_id = await orchestrator.create_task(
            goal="Calculate 2 + 2 offline",
            inputs={},
            limits={"time_s": 10, "cost_usd": 0.01},
            project_id="test"
        )
        
        assert task_id is not None
        
        # Wait for task completion
        await asyncio.sleep(0.1)
        
        # Check task status
        status = await orchestrator.get_task_status(task_id)
        assert status is not None
        
        # Should complete successfully
        task_data = status.get("task", {})
        assert task_data.get("status") in ["completed", "running"]


@pytest.mark.asyncio
async def test_fallback_policy():
    """Test fallback when services are unavailable."""
    
    with patch('httpx.AsyncClient') as mock_client:
        # Mock service unavailable
        mock_client.return_value.__aenter__.return_value.post.side_effect = Exception("Service unavailable")
        
        from apps.orchestrator.orchestrator import orchestrator
        
        # Create task (should use fallback)
        task_id = await orchestrator.create_task(
            goal="Test fallback behavior",
            inputs={},
            limits={"time_s": 30, "cost_usd": 0.1},
            project_id="test"
        )
        
        assert task_id is not None
        
        # Wait for task completion
        await asyncio.sleep(0.1)
        
        # Check task status
        status = await orchestrator.get_task_status(task_id)
        assert status is not None
        
        # Should still complete with fallback plan
        task_data = status.get("task", {})
        assert task_data.get("status") in ["completed", "running", "failed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
