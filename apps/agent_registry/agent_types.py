"""Agent Types and Capabilities Definitions."""

from typing import Any, Dict, List, Set
from dataclasses import dataclass
from enum import Enum


class AgentCapability(Enum):
    """Standard agent capabilities."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WEB_SEARCH = "web_search"
    DATA_PROCESSING = "data_processing"
    EXECUTION = "execution"
    TOOL_USAGE = "tool_usage"
    FILE_OPERATIONS = "file_operations"
    VALIDATION = "validation"
    CHECKING = "checking"
    QUALITY_ASSURANCE = "quality_assurance"
    LEARNING = "learning"
    ADAPTATION = "adaptation"
    PATTERN_RECOGNITION = "pattern_recognition"
    COORDINATION = "coordination"
    PLANNING = "planning"
    RESOURCE_MANAGEMENT = "resource_management"
    COMMUNICATION = "communication"
    NEGOTIATION = "negotiation"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE_THINKING = "creative_thinking"
    ETHICAL_REASONING = "ethical_reasoning"
    SAFETY_MONITORING = "safety_monitoring"


@dataclass
class AgentTypeDefinition:
    """Definition of an agent type."""
    type_id: str
    name: str
    description: str
    category: str  # "cognitive", "execution", "specialized"
    base_capabilities: Set[AgentCapability]
    optional_capabilities: Set[AgentCapability]
    resource_requirements: Dict[str, float]
    performance_characteristics: Dict[str, Any]
    specialization_domains: List[str]
    compatibility_requirements: Dict[str, Any]

    def get_all_capabilities(self) -> Set[AgentCapability]:
        """Get all possible capabilities for this agent type."""
        return self.base_capabilities.union(self.optional_capabilities)

    def validate_capabilities(self, capabilities: Set[str]) -> bool:
        """Validate that provided capabilities are valid for this type."""
        capability_enums = {AgentCapability(cap) for cap in capabilities}
        return capability_enums.issubset(self.get_all_capabilities())


# Predefined agent type definitions
AGENT_TYPE_DEFINITIONS = {
    # Cognitive Agents
    "researcher_advanced": AgentTypeDefinition(
        type_id="researcher_advanced",
        name="Advanced Researcher",
        description="High-capability research agent with advanced analysis and synthesis capabilities",
        category="cognitive",
        base_capabilities={
            AgentCapability.RESEARCH,
            AgentCapability.ANALYSIS,
            AgentCapability.WEB_SEARCH,
            AgentCapability.PATTERN_RECOGNITION,
            AgentCapability.PROBLEM_SOLVING
        },
        optional_capabilities={
            AgentCapability.DATA_PROCESSING,
            AgentCapability.CREATIVE_THINKING,
            AgentCapability.ETHICAL_REASONING
        },
        resource_requirements={
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu_memory_gb": 2,
            "storage_gb": 50
        },
        performance_characteristics={
            "expected_accuracy": 0.85,
            "response_time_seconds": 30,
            "context_window": 8192,
            "multilingual_support": True
        },
        specialization_domains=[
            "scientific_research",
            "market_analysis",
            "technical_documentation",
            "literature_review"
        ],
        compatibility_requirements={
            "min_model_size": "7B",
            "required_training": "domain_specific"
        }
    ),

    "reasoner_logical": AgentTypeDefinition(
        type_id="reasoner_logical",
        name="Logical Reasoner",
        description="Specialized in formal logic, mathematical reasoning, and analytical thinking",
        category="cognitive",
        base_capabilities={
            AgentCapability.ANALYSIS,
            AgentCapability.PROBLEM_SOLVING,
            AgentCapability.VALIDATION,
            AgentCapability.PATTERN_RECOGNITION,
            AgentCapability.DATA_PROCESSING
        },
        optional_capabilities={
            AgentCapability.MATHEMATICAL_MODELING,
            AgentCapability.STATISTICAL_ANALYSIS,
            AgentCapability.ALGORITHM_DESIGN
        },
        resource_requirements={
            "cpu_cores": 2,
            "memory_gb": 4,
            "gpu_memory_gb": 1,
            "storage_gb": 20
        },
        performance_characteristics={
            "expected_accuracy": 0.95,
            "response_time_seconds": 15,
            "mathematical_precision": 0.99,
            "logical_consistency": 0.98
        },
        specialization_domains=[
            "mathematical_reasoning",
            "logical_analysis",
            "algorithm_design",
            "formal_verification"
        ],
        compatibility_requirements={
            "min_model_size": "3B",
            "specialized_training": "mathematical"
        }
    ),

    "creator_creative": AgentTypeDefinition(
        type_id="creator_creative",
        name="Creative Generator",
        description="Specialized in creative content generation, ideation, and innovative solutions",
        category="cognitive",
        base_capabilities={
            AgentCapability.CREATIVE_THINKING,
            AgentCapability.PROBLEM_SOLVING,
            AgentCapability.ANALYSIS,
            AgentCapability.COMMUNICATION
        },
        optional_capabilities={
            AgentCapability.CONTENT_GENERATION,
            AgentCapability.IDEA_SYNTHESIS,
            AgentCapability.USER_EXPERIENCE_DESIGN
        },
        resource_requirements={
            "cpu_cores": 3,
            "memory_gb": 6,
            "gpu_memory_gb": 4,
            "storage_gb": 100
        },
        performance_characteristics={
            "expected_creativity_score": 0.8,
            "response_time_seconds": 45,
            "originality_index": 0.75,
            "coherence_score": 0.9
        },
        specialization_domains=[
            "content_creation",
            "product_design",
            "marketing_copy",
            "creative_problem_solving"
        ],
        compatibility_requirements={
            "min_model_size": "13B",
            "creative_training": True
        }
    ),

    # Execution Agents
    "executor_general": AgentTypeDefinition(
        type_id="executor_general",
        name="General Executor",
        description="Versatile execution agent capable of running various tools and workflows",
        category="execution",
        base_capabilities={
            AgentCapability.EXECUTION,
            AgentCapability.TOOL_USAGE,
            AgentCapability.FILE_OPERATIONS,
            AgentCapability.DATA_PROCESSING
        },
        optional_capabilities={
            AgentCapability.API_INTEGRATION,
            AgentCapability.AUTOMATION_SCRIPTING,
            AgentCapability.WORKFLOW_ORCHESTRATION
        },
        resource_requirements={
            "cpu_cores": 2,
            "memory_gb": 4,
            "gpu_memory_gb": 0,
            "storage_gb": 10
        },
        performance_characteristics={
            "success_rate": 0.9,
            "execution_speed": 0.8,
            "error_recovery": 0.85,
            "resource_efficiency": 0.9
        },
        specialization_domains=[
            "workflow_execution",
            "data_processing",
            "file_management",
            "tool_integration"
        ],
        compatibility_requirements={
            "sandbox_environment": True,
            "tool_registry_access": True
        }
    ),

    "executor_specialized": AgentTypeDefinition(
        type_id="executor_specialized",
        name="Specialized Executor",
        description="Domain-specific execution agent with deep expertise in particular tools/technologies",
        category="execution",
        base_capabilities={
            AgentCapability.EXECUTION,
            AgentCapability.TOOL_USAGE,
            AgentCapability.EXPERTISE_DOMAIN_SPECIFIC
        },
        optional_capabilities={
            AgentCapability.CODE_GENERATION,
            AgentCapability.DEBUGGING,
            AgentCapability.OPTIMIZATION
        },
        resource_requirements={
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu_memory_gb": 2,
            "storage_gb": 50
        },
        performance_characteristics={
            "domain_expertise_score": 0.95,
            "execution_accuracy": 0.92,
            "problem_solving_capability": 0.88,
            "adaptation_speed": 0.75
        },
        specialization_domains=[
            "software_development",
            "data_science",
            "system_administration",
            "devops_automation"
        ],
        compatibility_requirements={
            "domain_certifications": True,
            "specialized_tooling": True
        }
    ),

    # Specialized Agents
    "safety_monitor": AgentTypeDefinition(
        type_id="safety_monitor",
        name="Safety Monitor",
        description="Dedicated safety and security monitoring agent",
        category="specialized",
        base_capabilities={
            AgentCapability.SAFETY_MONITORING,
            AgentCapability.VALIDATION,
            AgentCapability.ETHICAL_REASONING,
            AgentCapability.RISK_ASSESSMENT
        },
        optional_capabilities={
            AgentCapability.ANOMALY_DETECTION,
            AgentCapability.COMPLIANCE_CHECKING,
            AgentCapability.INCIDENT_RESPONSE
        },
        resource_requirements={
            "cpu_cores": 2,
            "memory_gb": 4,
            "gpu_memory_gb": 1,
            "storage_gb": 25
        },
        performance_characteristics={
            "detection_accuracy": 0.95,
            "response_time_ms": 100,
            "false_positive_rate": 0.02,
            "coverage_completeness": 0.98
        },
        specialization_domains=[
            "security_monitoring",
            "compliance_auditing",
            "risk_assessment",
            "incident_detection"
        ],
        compatibility_requirements={
            "security_clearance": "high",
            "audit_logging": True,
            "real_time_processing": True
        }
    ),

    "coordinator_swarm": AgentTypeDefinition(
        type_id="coordinator_swarm",
        name="Swarm Coordinator",
        description="Advanced coordination agent for managing multi-agent swarms",
        category="specialized",
        base_capabilities={
            AgentCapability.COORDINATION,
            AgentCapability.PLANNING,
            AgentCapability.RESOURCE_MANAGEMENT,
            AgentCapability.COMMUNICATION,
            AgentCapability.NEGOTIATION
        },
        optional_capabilities={
            AgentCapability.CONFLICT_RESOLUTION,
            AgentCapability.LOAD_BALANCING,
            AgentCapability.PERFORMANCE_OPTIMIZATION
        },
        resource_requirements={
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu_memory_gb": 2,
            "storage_gb": 30
        },
        performance_characteristics={
            "coordination_efficiency": 0.9,
            "conflict_resolution_rate": 0.95,
            "resource_utilization": 0.88,
            "swarm_scalability": 0.85
        },
        specialization_domains=[
            "multi_agent_coordination",
            "resource_allocation",
            "conflict_management",
            "performance_optimization"
        ],
        compatibility_requirements={
            "multi_agent_experience": True,
            "negotiation_protocols": True,
            "real_time_communication": True
        }
    ),

    "learner_adaptive": AgentTypeDefinition(
        type_id="learner_adaptive",
        name="Adaptive Learner",
        description="Continuous learning agent capable of adapting to new domains and tasks",
        category="specialized",
        base_capabilities={
            AgentCapability.LEARNING,
            AgentCapability.ADAPTATION,
            AgentCapability.PATTERN_RECOGNITION,
            AgentCapability.ANALYSIS
        },
        optional_capabilities={
            AgentCapability.META_LEARNING,
            AgentCapability.CURRICULUM_LEARNING,
            AgentCapability.TRANSFER_LEARNING
        },
        resource_requirements={
            "cpu_cores": 6,
            "memory_gb": 16,
            "gpu_memory_gb": 8,
            "storage_gb": 200
        },
        performance_characteristics={
            "learning_speed": 0.8,
            "adaptation_accuracy": 0.85,
            "knowledge_retention": 0.9,
            "generalization_capability": 0.75
        },
        specialization_domains=[
            "continuous_learning",
            "domain_adaptation",
            "skill_acquisition",
            "knowledge_integration"
        ],
        compatibility_requirements={
            "reinforcement_learning": True,
            "meta_learning_algorithms": True,
            "large_knowledge_base": True
        }
    )
}


def get_agent_type_definition(type_id: str) -> AgentTypeDefinition:
    """Get agent type definition by ID."""
    if type_id not in AGENT_TYPE_DEFINITIONS:
        raise ValueError(f"Unknown agent type: {type_id}")
    return AGENT_TYPE_DEFINITIONS[type_id]


def list_available_agent_types(category: str = None) -> List[AgentTypeDefinition]:
    """List available agent types, optionally filtered by category."""
    types = list(AGENT_TYPE_DEFINITIONS.values())

    if category:
        types = [t for t in types if t.category == category]

    return types


def validate_agent_capabilities(type_id: str, capabilities: Set[str]) -> bool:
    """Validate that capabilities are compatible with agent type."""
    try:
        agent_type = get_agent_type_definition(type_id)
        return agent_type.validate_capabilities(capabilities)
    except ValueError:
        return False


def get_compatible_agent_types(capabilities: Set[str]) -> List[AgentTypeDefinition]:
    """Find agent types compatible with given capabilities."""
    compatible_types = []

    for agent_type in AGENT_TYPE_DEFINITIONS.values():
        if agent_type.validate_capabilities(capabilities):
            compatible_types.append(agent_type)

    return compatible_types


def calculate_agent_resource_score(type_id: str, available_resources: Dict[str, float]) -> float:
    """Calculate how well available resources match agent type requirements."""

    try:
        agent_type = get_agent_type_definition(type_id)
        requirements = agent_type.resource_requirements

        total_score = 0.0
        total_weight = 0.0

        for resource, required in requirements.items():
            available = available_resources.get(resource, 0.0)
            weight = 1.0  # Could be made configurable

            if available >= required:
                score = 1.0
            else:
                score = available / required if required > 0 else 0.0

            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    except ValueError:
        return 0.0
