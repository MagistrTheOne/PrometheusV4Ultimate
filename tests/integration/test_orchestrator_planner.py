"""Integration tests for Orchestrator + Planner."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from apps.orchestrator.orchestrator import orchestrator
from apps.planner.planner import planner


@pytest.mark.asyncio
async def test_orchestrator_planner_integration():
    """Test Orchestrator integration with Planner."""
    
    # Mock the planner HTTP call
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "plan_id": "test-plan-001",
            "steps": [
                {
                    "name": "test_step",
                    "skill_name": "math_calc",
                    "inputs": {"expression": "1+1"},
                    "description": "Test calculation"
                }
            ],
            "budget": {"total_cost_usd": 0.001, "total_time_ms": 1000},
            "estimated_duration_ms": 1000
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        # Create task
        task_id = await orchestrator.create_task(
            goal="test integration",
            inputs={"test": "data"},
            limits={"time_s": 60, "cost_usd": 0.1},
            project_id="test"
        )
        
        assert task_id is not None
        
        # Wait a bit for task to process
        await asyncio.sleep(0.1)
        
        # Check task status
        status = await orchestrator.get_task_status(task_id)
        assert status is not None
        assert "task" in status
        assert "steps" in status
        assert "events" in status


@pytest.mark.asyncio
async def test_planner_plan_creation():
    """Test Planner plan creation."""
    
    plan = await planner.create_plan(
        goal="create CSV report",
        inputs={"csv_file": "data.csv"},
        limits={"time_s": 30, "cost_usd": 0.05},
        project_id="test",
        policy="auto"
    )
    
    assert plan is not None
    assert "id" in plan
    assert "steps" in plan
    assert "budget" in plan
    assert "estimated_duration_ms" in plan
    
    # Check that plan has reasonable structure
    assert len(plan["steps"]) > 0
    assert plan["budget"]["total_cost_usd"] > 0
    assert plan["estimated_duration_ms"] > 0


@pytest.mark.asyncio
async def test_planner_skill_metrics():
    """Test Planner skill metrics."""
    
    # Update metrics
    planner.update_skill_metrics("math_calc", 500, 0.0005, True)
    
    # Check metrics were updated
    metrics = planner.skill_metrics["math_calc"]
    assert metrics.avg_latency_ms > 0
    assert metrics.avg_cost_usd > 0
    assert metrics.success_rate > 0


@pytest.mark.asyncio
async def test_planner_policy_selection():
    """Test Planner policy selection."""
    
    # Test different policies
    policies = ["tiny", "base", "balanced", "ultra", "mega", "auto"]
    
    for policy in policies:
        plan = await planner.create_plan(
            goal="test policy",
            inputs={},
            limits={"time_s": 60, "cost_usd": 1.0},
            project_id="test",
            policy=policy
        )
        
        assert plan is not None
        assert plan["policy"] == policy or policy == "auto"


if __name__ == "__main__":
    pytest.main([__file__])
