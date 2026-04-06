# sdd_fullcycle_langchain.py
"""
Spec-Driven Development (SDD) full cycle with LangGraph + LangChain.

Pipeline (end-to-end):
    idea
      → specify_node        : generate spec_md (Markdown spec)
      → clarify_node        : generate clarifications + best-practice answers
      → plan_node           : produce impl_plan_md + tasks_md
      → implement_node      : execute DEV tasks via a generic full_agent
      → test_node           : generate & run tests from spec_md
      → review_gate_node    : decide "done" or loop back to implement

This file is GENERIC:
- It does not hardcode any domain (e-commerce, APIs, etc.).
- You can pass any high-level `idea` string.
- The plan/tasks/implementation will be shaped by the spec for that idea.
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Annotated, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────
# 1. SDD State definition
# ─────────────────────────────────────────────────────────────

class SDDState(TypedDict):
    """
    Shared state that flows through the SDD LangGraph.

    Fields:
        idea:
            High-level idea / requirement description from the user.
        spec_md:
            Generated specification in Markdown (source of truth).
        clarifications:
            List of clarification questions about the spec.
        clarification_answers:
            List of answers to those questions (here, auto-generated "best practices").
        impl_plan_md:
            Implementation plan in Markdown (plan.md).
        tasks_md:
            Task checklist in Markdown (tasks.md), grouped by BA/DEV/QA/RELEASE.
        dev_tasks:
            Extracted DEV-* tasks from tasks_md.
        code_changes:
            Map DEV task -> summary of what full_agent did.
        test_code_py:
            Generated Python test code string.
        test_results:
            Human-readable results from running tests.
        status:
            Overall SDD status: "testing", "needs_fix", or "done".
    """
    idea: str
    spec_md: str
    clarifications: List[str]
    clarification_answers: List[str]
    impl_plan_md: str
    tasks_md: str
    dev_tasks: List[str]
    code_changes: Dict[str, str]
    test_code_py: str
    test_results: List[str]
    status: str  # "testing" | "needs_fix" | "done"


# ─────────────────────────────────────────────────────────────
# 2. LLM clients for each SDD phase
# ─────────────────────────────────────────────────────────────

# Separate LLM instances per phase (spec, clarify, plan, test).
# You can swap models or temperatures independently if needed.
llm_spec = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_clarify = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_plan = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_test = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─────────────────────────────────────────────────────────────
# 3. Node: specify_node (idea → spec_md)
# ─────────────────────────────────────────────────────────────

def specify_node(state: SDDState) -> dict:
    """
    Turn a high-level idea into a structured Markdown specification (spec_md).

    Uses SDD-style instructions:
    - Overview
    - User Personas
    - User Stories (with Gherkin-style acceptance criteria)
    - Functional & Non-functional requirements
    - Edge cases & Open questions
    """
    idea = state["idea"]

    system_msg = (
        "You are a senior product engineer practicing Spec-Driven Development.\n"
        "Given a high-level idea, produce a SPECIFICATION in Markdown.\n"
        "The spec is for AI-assisted implementation later, so it must be:\n"
        "- concrete, testable, and unambiguous\n"
        "- focused on WHAT to build, not HOW.\n"
        "Follow this structure:\n"
        "1. Overview\n"
        "2. User Personas\n"
        "3. User Stories (with Gherkin-style Acceptance Criteria per story)\n"
        "4. Functional Requirements\n"
        "5. Non-Functional Requirements\n"
        "6. Edge Cases & Constraints\n"
        "7. Open Questions\n"
        "Use Given/When/Then format for acceptance criteria."
    )

    user_msg = (
        "High-level idea:\n"
        f"{idea}\n\n"
        "Context: Generate a generic spec for this idea (do NOT assume any fixed domain).\n"
        "Target: should be usable as the source of truth for implementation."
    )

    response = llm_spec.invoke(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    )

    spec_md = response.content.strip()
    return {"spec_md": spec_md}


# ─────────────────────────────────────────────────────────────
# 4. Node: clarify_node (spec_md → clarifications + answers)
# ─────────────────────────────────────────────────────────────

def clarify_node(state: SDDState) -> dict:
    """
    Clarify the specification using SDD best practices.

    This node does TWO things:
      1) Generates 3–5 clarification questions about the spec.
      2) Immediately answers them with best-practice defaults.

    Output:
        clarifications: list of questions
        clarification_answers: list of answers (same length as clarifications)
    """
    spec_md = state["spec_md"]

    system_msg = (
        "You are an expert product/engineering lead practicing Spec-Driven Development.\n"
        "Your job is to clarify specs using industry best practices.\n"
        "Step 1: Read the spec and identify 3–5 important clarification questions.\n"
        "Step 2: For EACH question, provide a concise, best-practice answer yourself.\n"
        "Output format:\n"
        "QUESTIONS:\n"
        "1. ...\n"
        "2. ...\n"
        "\n"
        "ANSWERS:\n"
        "1. ...\n"
        "2. ...\n"
        "Answers should be realistic defaults that a strong product/engineering team\n"
        "would choose if the spec writer didn't say otherwise."
    )

    user_msg = (
        "Here is the current spec.md.\n\n"
        "SPEC START\n"
        f"{spec_md}\n"
        "SPEC END\n\n"
        "Generate questions AND best-practice answers in the format described."
    )

    response = llm_clarify.invoke(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    )

    text = response.content.strip()

    questions: List[str] = []
    answers: List[str] = []

    # Split into QUESTIONS / ANSWERS sections
    if "ANSWERS:" in text:
        q_part, a_part = text.split("ANSWERS:", 1)
    else:
        q_part, a_part = text, ""

    # Extract question lines
    if "QUESTIONS:" in q_part:
        q_lines = q_part.split("QUESTIONS:", 1)[1].strip().split("\n")
    else:
        q_lines = q_part.strip().split("\n")

    for line in q_lines:
        line = line.strip()
        if not line:
            continue
        # Remove leading "1. " or "2. " etc.
        if line[0].isdigit() and "." in line:
            line = line.split(".", 1)[1].strip()
        questions.append(line)

    # Extract answer lines
    a_lines = a_part.strip().split("\n") if a_part.strip() else []
    for line in a_lines:
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and "." in line:
            line = line.split(".", 1)[1].strip()
        answers.append(line)

    # Ensure lengths align (truncate to min length)
    n = min(len(questions), len(answers))
    questions = questions[:n]
    answers = answers[:n]

    return {
        "clarifications": questions,
        "clarification_answers": answers,
    }


# ─────────────────────────────────────────────────────────────
# 5. Node: plan_node (spec_md + answers → impl_plan_md + tasks_md)
# ─────────────────────────────────────────────────────────────

def plan_node(state: SDDState) -> dict:
    """
    Turn the clarified spec into:
      - impl_plan_md : high-level implementation plan (plan.md)
      - tasks_md     : detailed tasks grouped by BA/DEV/QA/RELEASE (tasks.md)

    The node merges:
      - spec_md
      - clarifications
      - clarification_answers

    Then asks the LLM to output two sections:
      === PLAN_MD ===
      (Markdown plan)
      === TASKS_MD ===
      (Markdown checklist)
    """
    spec_md = state["spec_md"]
    clarifications = state.get("clarifications", [])
    answers = state.get("clarification_answers", [])

    # Merge clarifications+answers into the spec so the planner sees the full context.
    merged_spec = spec_md
    if clarifications and answers:
        merged_spec += "\n\n## Clarification Answers\n"
        for i, (q, a) in enumerate(zip(clarifications, answers), 1):
            merged_spec += f"- Q{i}: {q}\n  A{i}: {a}\n"

    system_msg = (
        "You are a senior tech lead practicing Spec-Driven Development.\n"
        "Given a clarified feature spec, you produce two Markdown artifacts:\n"
        "1) plan.md – implementation plan\n"
        "2) tasks.md – detailed checklist of tasks derived from the plan.\n\n"
        "Methodology:\n"
        "- Analyze user stories, FR, NFR, edge cases.\n"
        "- Create a concise plan grouped by phases: BA, DEV, QA, RELEASE.\n"
        "- Break the plan into small, independent tasks an AI agent can execute.\n"
        "- Each task MUST be:\n"
        "  - implementable and testable in isolation,\n"
        "  - mapped to at least one user story / requirement.\n"
        "Output format:\n"
        "=== PLAN_MD ===\n"
        "(Markdown plan)\n"
        "=== TASKS_MD ===\n"
        "(Markdown checklist)"
    )

    user_msg = (
        "CLARIFIED SPEC BELOW:\n"
        "--------------------------------\n"
        f"{merged_spec}\n"
        "--------------------------------\n"
    )

    response = llm_plan.invoke(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    )

    full = response.content.strip()
    plan_part = ""
    tasks_part = ""

    if "=== PLAN_MD ===" in full and "=== TASKS_MD ===" in full:
        _, rest = full.split("=== PLAN_MD ===", 1)
        plan_part, maybe_tasks = rest.split("=== TASKS_MD ===", 1)
        plan_part = plan_part.strip()
        tasks_part = maybe_tasks.strip()
    else:
        # Fallback if the model doesn't follow the format
        plan_part = full
        tasks_part = ""

    return {
        "impl_plan_md": plan_part,
        "tasks_md": tasks_part,
    }


# ─────────────────────────────────────────────────────────────
# 6. full_agent: generic ReAct-style agent for implement_node
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    State for the inner "implementation agent".

    messages:
        Conversation history used by the LLM + tools.
        Annotated with add_messages to append instead of overwrite.
    """
    messages: Annotated[List, add_messages]


@tool
def search_web(query: str) -> str:
    """Mock web search tool (no real network calls)."""
    return f"[MOCK SEARCH] You asked: {query}"


@tool
def run_python(code: str) -> str:
    """
    Execute Python code in a simple REPL (demo only).

    NOTE: In real applications, this must be sandboxed properly.
    """
    local_vars: Dict[str, object] = {}
    try:
        exec(code, {}, local_vars)
        return f"Executed successfully. Locals: {local_vars}"
    except Exception as e:
        return f"Error while executing code: {e!r}"


@tool
def calculator(expr: str) -> str:
    """Evaluate a math expression using numexpr."""
    import numexpr
    try:
        val = numexpr.evaluate(expr).item()
        return str(val)
    except Exception as e:
        return f"Error while evaluating expression: {e!r}"


# Tools available to the inner agent.
tools = [search_web, run_python, calculator]

# LLM for the implementation agent, with tools bound for tool_calls.
llm_agent = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm_agent.bind_tools(tools)


def call_model(state: AgentState) -> dict:
    """
    Inner agent node:
      - Takes the current message history.
      - Calls the tool-enabled LLM.
      - Appends the AIMessage to the history.
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Build the inner agent graph: agent ↔ tools loop (ReAct pattern).
agent_graph = StateGraph(AgentState)
agent_graph.add_node("agent", call_model)
agent_graph.add_node("tools", ToolNode(tools))
agent_graph.add_edge(START, "agent")
agent_graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
agent_graph.add_edge("tools", "agent")

# In-memory checkpointer for per-thread conversations.
agent_memory = MemorySaver()
full_agent = agent_graph.compile(checkpointer=agent_memory)


# ─────────────────────────────────────────────────────────────
# 7. Helper: extract DEV tasks from tasks_md
# ─────────────────────────────────────────────────────────────

def extract_dev_tasks(tasks_md: str) -> List[str]:
    """
    Parse tasks_md and extract DEV-* tasks.

    Example input line:
        - [ ] DEV-1: Build header (links: US-1, FR-2)
    We return:
        "DEV-1: Build header (links: US-1, FR-2)"
    """
    dev_tasks: List[str] = []
    for line in tasks_md.splitlines():
        line = line.strip()
        if not line.startswith("- [ ] DEV-"):
            continue
        # Drop the leading "- [ ] "
        task_text = line[len("- [ ] "):].strip()
        dev_tasks.append(task_text)
    return dev_tasks


# ─────────────────────────────────────────────────────────────
# 8. Node: implement_node (tasks_md → dev_tasks + code_changes)
# ─────────────────────────────────────────────────────────────

def implement_node(state: SDDState) -> dict:
    """
    For each DEV-* task, call the `full_agent` as a subgraph.

    Workflow:
      - Extract DEV tasks from tasks_md.
      - For each task, build a prompt that includes:
          - The spec (spec_md).
          - The specific DEV task.
      - Call full_agent.invoke(...) with a per-task thread_id.
      - Store the assistant's summary of changes in code_changes[task].
    """
    tasks_md = state["tasks_md"]
    dev_tasks = extract_dev_tasks(tasks_md)
    code_changes: Dict[str, str] = {}

    for task in dev_tasks:
        # Example DEV task: "DEV-1: Build header for home screen (links: US-1, FR-2)"
        dev_id = task.split(":", 1)[0]  # "DEV-1"
        thread_id = f"sdd-dev-{dev_id}"

        user_prompt = (
            "You are a coding agent implementing a feature according to a spec.\n\n"
            f"SPEC (spec_md):\n{state['spec_md']}\n\n"
            f"IMPLEMENT THIS TASK ONLY:\n{task}\n\n"
            "Constraints:\n"
            "- Make minimal, focused changes.\n"
            "- Prefer creating or editing a single file for this task.\n"
            "- At the end, summarize which file(s) you changed and what you did.\n"
        )

        result = full_agent.invoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            {"configurable": {"thread_id": thread_id}},
        )
        last_msg = result["messages"][-1]
        code_changes[task] = last_msg.content

    return {
        "dev_tasks": dev_tasks,
        "code_changes": code_changes,
    }


# ─────────────────────────────────────────────────────────────
# 9. Node: test_node (spec_md → test_code_py + test_results + status)
# ─────────────────────────────────────────────────────────────

def test_node(state: SDDState) -> dict:
    """
    Generate Python tests from spec_md and execute them via run_python.

    This is a lightweight test generator:
      - It asks the LLM for pytest-style functions.
      - It strips any ``` fences that might appear.
      - It runs the code in-process via run_python (demo only).
      - It marks the state.status as "done" or "needs_fix" based on output.
    """
    spec_md = state["spec_md"]

    system_msg_content = (
        "You are a QA engineer practicing Spec-Driven Development.\n"
        "Given a feature spec with acceptance criteria, generate Python tests.\n"
        "Constraints:\n"
        "- Use pytest-style functions (test_...).\n"
        "- Focus on logical checks (helpers/config), NOT full browser automation.\n"
        "- Keep everything in a single Python file.\n"
        "- Return ONLY raw Python code, with NO markdown fences and NO ```."
    )

    user_msg_content = (
        "SPEC (spec.md):\n"
        "----------------------\n"
        f"{spec_md}\n"
        "----------------------\n\n"
        "Generate pytest-style tests that validate:\n"
        "- At least 3 critical acceptance criteria from the spec.\n"
        "- Edge cases for empty data or failed dependencies.\n"
        "Return ONLY Python code."
    )

    messages = [
        SystemMessage(content=system_msg_content),
        HumanMessage(content=user_msg_content),
    ]

    response = llm_test.invoke(messages)
    raw = response.content.strip()

    # Strip ``` fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else ""
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    test_code = raw.strip()

    try:
        test_output = run_python.run(test_code)
        test_results = [f"OK: tests executed\n{test_output}"]
        upper = test_output.upper()
        status = "done" if "FAIL" not in upper and "ERROR" not in upper else "needs_fix"
    except Exception as e:
        test_results = [f"ERROR while running tests: {e!r}"]
        status = "needs_fix"

    return {
        "test_code_py": test_code,
        "test_results": test_results,
        "status": status,
    }


# ─────────────────────────────────────────────────────────────
# 10. Node: review_gate_node + router
# ─────────────────────────────────────────────────────────────

def review_gate_node(state: SDDState) -> dict:
    """
    Simple review gate.

    Logic:
      - If any test result mentions FAIL/ERROR/ASSERT, mark as needs_fix.
      - Otherwise, mark as done.

    The router will decide whether to END or go back to implement_node.
    """
    results = state.get("test_results", [])
    text = "\n".join(results).upper()

    if "FAIL" in text or "ERROR" in text or "ASSERT" in text:
        status = "needs_fix"
    else:
        status = "done"

    return {"status": status}


def route_after_review(state: SDDState) -> Literal["implement", "END"]:
    """
    Conditional edge function for the 'review' node.

    If status == "done"  → END (SDD cycle completed).
    Otherwise            → "implement" (loop back to implement_node).
    """
    if state["status"] == "done":
        return "END"
    return "implement"


# ─────────────────────────────────────────────────────────────
# 11. Build SDD graph
# ─────────────────────────────────────────────────────────────

def build_sdd_app():
    """
    Build and compile the full SDD StateGraph.

    Nodes:
      "specify"   : idea → spec_md
      "clarify"   : spec_md → clarifications + clarification_answers
      "plan"      : spec_md+answers → impl_plan_md + tasks_md
      "implement" : tasks_md → dev_tasks + code_changes (via full_agent)
      "test"      : spec_md → test_code_py + test_results + status
      "review"    : test_results → status, route to END or implement

    Returns:
        app: compiled LangGraph application you can invoke/stream.
    """
    graph = StateGraph(SDDState)
    graph.add_node("specify", specify_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("implement", implement_node)
    graph.add_node("test", test_node)
    graph.add_node("review", review_gate_node)

    # Linear flow: specify → clarify → plan → implement → test → review
    graph.add_edge(START, "specify")
    graph.add_edge("specify", "clarify")
    graph.add_edge("clarify", "plan")
    graph.add_edge("plan", "implement")
    graph.add_edge("implement", "test")
    graph.add_edge("test", "review")

    # Conditional edge from review to either END or back to implement
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "implement": "implement",
            "END": END,
        },
    )

    app = graph.compile(checkpointer=MemorySaver())
    return app


# ─────────────────────────────────────────────────────────────
# 12. Example usage (run this file directly)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build the SDD app and define a thread_id for stateful runs.
    app = build_sdd_app()
    config = {"configurable": {"thread_id": "sdd-demo-1"}}

    # Any idea works here – you can swap this out.
    idea = "Home screen for displaying an e-commerce web page for coffee products"

    # Initial state; most fields are empty strings/lists and will be filled by the graph.
    state_in: SDDState = {
        "idea": idea,
        "spec_md": "",
        "clarifications": [],
        "clarification_answers": [],
        "impl_plan_md": "",
        "tasks_md": "",
        "dev_tasks": [],
        "code_changes": {},
        "test_code_py": "",
        "test_results": [],
        "status": "",
    }

    result = app.invoke(state_in, config)

    print("\n=== FINAL STATE ===")
    print("Status:", result.get("status"))
    print("\nSPEC_MD (first 400 chars):")
    print(result["spec_md"][:400], "...\n")
    print("TASKS_MD (first 400 chars):")
    print(result["tasks_md"][:400], "...\n")
    print("DEV TASKS:", result.get("dev_tasks", []))
    print("TEST RESULTS:", result.get("test_results", []))
