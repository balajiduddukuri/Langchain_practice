
"""
AI SDLC using LangGraph Swarm Pattern
----------------------------------
This module demonstrates how an end-to-end AI SDLC (Software Development Life Cycle)
can be modelled using swarm-style multi-agent architecture.

Agents represent SDLC roles and dynamically hand off control based on context.
"""

# ==============================
# 1. Imports
# ==============================
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph_swarm import create_swarm, create_handoff_tool
from langgraph.checkpoint.memory import InMemorySaver

# ==============================
# 2. Shared LLM
# ==============================
model = ChatOpenAI(model="gpt-4o")

# ==============================
# 3. SDLC Agents
# ==============================

# --- Requirements Agent ---
requirements_agent = create_agent(
    model,
    tools=[
        create_handoff_tool(
            agent_name="ArchitectureAgent",
            description="Move to system / solution design"
        )
    ],
    system_prompt="""
    You are a Business Analyst.
    Responsibilities:
    - Understand business goals
    - Clarify requirements
    - Define acceptance criteria
    - Decide when requirements are ready for design
    """,
    name="RequirementsAgent",
)


# --- Architecture / Design Agent ---
architecture_agent = create_agent(
    model,
    tools=[
        create_handoff_tool(
            agent_name="DevelopmentAgent",
            description="Move to implementation and coding"
        ),
        create_handoff_tool(
            agent_name="RequirementsAgent",
            description="Return to requirements for clarification"
        )
    ],
    system_prompt="""
    You are a Solution Architect.
    Responsibilities:
    - Define system architecture
    - Choose patterns, tools, and integrations
    - Validate non-functional requirements
    """,
    name="ArchitectureAgent",
)


# --- Development Agent ---
development_agent = create_agent(
    model,
    tools=[
        create_handoff_tool(
            agent_name="TestingAgent",
            description="Move to testing and validation"
        ),
        create_handoff_tool(
            agent_name="ArchitectureAgent",
            description="Return for design changes"
        )
    ],
    system_prompt="""
    You are a Software Engineer.
    Responsibilities:
    - Write production-ready code
    - Follow architecture guidelines
    - Implement features and fixes
    """,
    name="DevelopmentAgent",
)


# --- Testing / QA Agent ---
testing_agent = create_agent(
    model,
    tools=[
        create_handoff_tool(
            agent_name="ReleaseAgent",
            description="Move to deployment and release"
        ),
        create_handoff_tool(
            agent_name="DevelopmentAgent",
            description="Return bugs to development"
        )
    ],
    system_prompt="""
    You are a QA Engineer.
    Responsibilities:
    - Validate functionality
    - Check edge cases and regressions
    - Approve or reject release readiness
    """,
    name="TestingAgent",
)


# --- Release / Ops Agent ---
release_agent = create_agent(
    model,
    tools=[
        create_handoff_tool(
            agent_name="RequirementsAgent",
            description="Start next iteration / feedback loop"
        )
    ],
    system_prompt="""
    You are a DevOps / Release Manager.
    Responsibilities:
    - Deployment strategy
    - Monitoring and rollback
    - Collect production feedback
    """,
    name="ReleaseAgent",
)

# ==============================
# 4. Create Swarm SDLC Workflow
# ==============================
workflow = create_swarm(
    agents=[
        requirements_agent,
        architecture_agent,
        development_agent,
        testing_agent,
        release_agent,
    ],
    default_active_agent="RequirementsAgent"
)

# ==============================
# 5. Compile with Memory
# ==============================
checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ==============================
# 6. Example Invocation
# ==============================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "ai_sdlc_demo"}}

    response = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "We need an AI system to automate contract review for compliance."
                }
            ]
        },
        config
    )

    print("
--- SDLC Swarm Output ---")
    print(response)
