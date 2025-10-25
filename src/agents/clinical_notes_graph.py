"""
Clinical Notes Multi-Agent Processing Graph.

This module implements the LangGraph orchestration for processing clinical notes
through multiple specialized agents with iterative quality refinement.

Architecture:
    Main Orchestrator
    ├── ToC Subagent (section detection)
    ├── Summary Subagent (clinical extraction)
    ├── Recommendation Subagent (treatment plan)
    └── Evaluator Agent (quality validation)

Flow:
    Input → Parallel Processing (ToC, Summary, Recommendations)
          → Evaluation → Pass: Complete / Fail: Refine (max 5 iterations)
"""

import os
import sys
from pathlib import Path
from typing import Literal, Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env file
load_dotenv()

# Add parent directories to path for imports
_current_dir = Path(__file__).parent
_src_dir = _current_dir.parent
_project_root = _src_dir.parent

# Import state model and logger
sys.path.insert(0, str(_src_dir))
from models.graph_state import ClinicalNoteState
from utils.agent_logger import AgentLogger

# ============================================================================
# CONFIGURATION
# ============================================================================

# Agent prompts directory
# Resolve absolute path to handle different execution contexts
_file_path = Path(__file__).resolve()
AGENTS_DIR = _file_path.parent.parent.parent / ".agent" / "agents"

# Skills directory
SKILLS_DIR = _file_path.parent.parent.parent / ".agent" / "skills"

# Verify directories exist
if not AGENTS_DIR.exists():
    raise FileNotFoundError(f"Agents directory not found: {AGENTS_DIR}")
if not SKILLS_DIR.exists():
    raise FileNotFoundError(f"Skills directory not found: {SKILLS_DIR}")

# Ollama model configuration
LLM_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
LLM_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Iteration limits
MAX_ITERATIONS = 2

# ============================================================================
# SKILL TOOL WRAPPERS
# ============================================================================
# Each skill in .agent/skills/ is wrapped as a LangChain tool
# Skills are Python modules with isolated .venv environments

import json
from langchain_core.tools import tool

# Shared client instances (initialized on first use)
_ollama_client = None
_chroma_client = None


def get_ollama_client():
    """Get or create Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        sys.path.insert(0, str(SKILLS_DIR / "ollama-client"))
        from ollama_client import OllamaClient
        _ollama_client = OllamaClient()
    return _ollama_client


def get_chroma_client():
    """Get or create ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        sys.path.insert(0, str(SKILLS_DIR / "chroma-client"))
        from chroma_client import ChromaClient
        _chroma_client = ChromaClient()
    return _chroma_client


@tool
def section_detection_tool(note_text: str) -> str:
    """
    Detect sections in clinical note using regex patterns.

    Args:
        note_text: Clinical note text to analyze

    Returns:
        JSON string with detected sections (title, start_offset, end_offset, confidence)
    """
    sys.path.insert(0, str(SKILLS_DIR / "section-detection"))
    from section_detection import SectionDetector

    detector = SectionDetector(get_ollama_client())
    sections = detector.detect_sections(note_text)

    return json.dumps(sections, indent=2)


@tool
def contextual_chunking_tool(document_text: str, doc_id: str) -> str:
    """
    Chunk document with LLM-generated contextual summaries for improved RAG retrieval.

    Args:
        document_text: Full clinical note text
        doc_id: Document identifier (for chunk IDs)

    Returns:
        JSON string with enriched chunks (id, original_text, context, enriched_text, offsets)
    """
    sys.path.insert(0, str(SKILLS_DIR / "contextual-chunking"))
    from contextual_chunking import ContextualChunker

    chunker = ContextualChunker(get_ollama_client())
    chunks = chunker.chunk_with_context(document_text, doc_id)

    return json.dumps(chunks, indent=2)


@tool
def rag_retrieval_tool(collection_name: str, query: str, n_results: int = 5) -> str:
    """
    Retrieve relevant passages from ChromaDB using hybrid search.

    Args:
        collection_name: ChromaDB collection to query
        query: Search query
        n_results: Number of results to return (default: 5)

    Returns:
        JSON string with retrieved passages (text, score, offsets)
    """
    sys.path.insert(0, str(SKILLS_DIR / "rag-retrieval"))
    from rag_retrieval import RAGRetriever

    retriever = RAGRetriever(get_chroma_client(), collection_name)
    results = retriever.retrieve(query, n_results)

    return json.dumps(results, indent=2)


@tool
def citation_extraction_tool(claim_text: str, source_text: str, start_offset: int, end_offset: int) -> str:
    """
    Create citation with Jaccard overlap validation.

    Args:
        claim_text: The claim being cited
        source_text: Source text from clinical note
        start_offset: Source start character position
        end_offset: Source end character position

    Returns:
        JSON string with citation (claim_text, source_text, offsets, jaccard_overlap, confidence)
    """
    sys.path.insert(0, str(SKILLS_DIR / "citation-extraction"))
    from citation_extraction import CitationExtractor

    extractor = CitationExtractor()
    citation = extractor.extract_citation(claim_text, source_text, start_offset, end_offset)

    return json.dumps(citation, indent=2)


@tool
def json_validation_tool(json_output: str, schema_name: str) -> str:
    """
    Validate JSON output against a Pydantic schema.

    Args:
        json_output: JSON string or LLM output containing JSON
        schema_name: Schema to validate against (toc, summary, plan, evaluator)

    Returns:
        JSON string with validation result (valid: bool, errors: List[str], validated_data: Dict)
    """
    sys.path.insert(0, str(SKILLS_DIR / "json-validation"))
    from json_validation import JSONValidator

    # Import appropriate schema
    schema_map = {
        "toc": "TableOfContents",
        "summary": "ClinicalSummary",
        "plan": "TreatmentPlan",
        "evaluator": "EvaluatorOutput"
    }

    if schema_name not in schema_map:
        return json.dumps({
            "valid": False,
            "errors": [f"Unknown schema: {schema_name}"],
            "validated_data": None
        })

    # For now, just validate it's valid JSON
    # TODO: Load actual Pydantic schemas from src/models/
    result = JSONValidator.validate_json_string(json_output, dict)

    return json.dumps(result, indent=2)


# Tool lists for each agent
SKILL_TOOLS = {
    "toc_tools": [
        section_detection_tool,
        json_validation_tool,
    ],
    "summary_tools": [
        contextual_chunking_tool,
        rag_retrieval_tool,
        citation_extraction_tool,
        json_validation_tool,
    ],
    "recommendation_tools": [
        rag_retrieval_tool,
        citation_extraction_tool,
        json_validation_tool,
    ],
    "evaluator_tools": [
        citation_extraction_tool,
        json_validation_tool,
    ],
}

# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def get_llm() -> ChatOllama:
    """
    Initialize Ollama client with JSON mode enabled.

    Returns:
        ChatOllama: Configured LLM instance with JSON output
    """
    return ChatOllama(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        temperature=0.1,  # Low temperature for clinical accuracy
        format="json",  # Force JSON output mode
    )


# ============================================================================
# AGENT PROMPT LOADING
# ============================================================================

def load_agent_prompt(agent_name: str) -> str:
    """
    Load agent instruction prompt from .agent/agents/ directory.

    Args:
        agent_name: Agent filename without .md extension
            (e.g., "toc-subagent", "summary-subagent")

    Returns:
        Prompt text as system message content

    Raises:
        FileNotFoundError: If agent prompt file doesn't exist
    """
    prompt_file = AGENTS_DIR / f"{agent_name}.md"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Agent prompt not found: {prompt_file}")

    return prompt_file.read_text(encoding="utf-8")


def extract_json_from_agent_output(result: Dict[str, Any]) -> str:
    """
    Extract JSON from agent's final message output.

    Args:
        result: Agent invocation result containing messages

    Returns:
        Extracted JSON string or error placeholder
    """
    messages = result.get("messages", [])
    if not messages:
        return '{"error": "No messages in agent output"}'

    # Get last message content
    last_message = messages[-1]
    content = getattr(last_message, "content", "")

    # Try to extract JSON from the content
    sys.path.insert(0, str(SKILLS_DIR / "json-validation"))
    from json_validation import JSONValidator

    json_str = JSONValidator.extract_json_from_text(content)

    if json_str is None:
        # If no JSON found, return the raw content wrapped in error
        return json.dumps({"error": "No JSON found in output", "raw_output": content})

    return json_str


# ============================================================================
# AGENT NODE FUNCTIONS
# ============================================================================
# Each agent is a graph node that processes the state and returns updates

def toc_agent_node(state: ClinicalNoteState) -> Dict[str, Any]:
    """
    ToC Subagent: Generate Table of Contents with section detection.

    Args:
        state: Current graph state with clinical note text

    Returns:
        State updates with toc_output (JSON string)
    """
    # Initialize logger
    session_id = state.get("session_id", "unknown")
    logger = AgentLogger(session_id)

    logger.log_message("toc_agent", "Starting Table of Contents generation")

    llm = get_llm()
    prompt = load_agent_prompt("toc-subagent")

    # Create agent with tools
    agent = create_react_agent(
        model=llm,
        tools=SKILL_TOOLS["toc_tools"],
        prompt=SystemMessage(content=prompt),
    )

    # Prepare input message
    input_message = HumanMessage(
        content=f"Generate Table of Contents for this clinical note:\n\n{state['clinical_note_text']}"
    )

    # Log input
    logger.log_message("toc_agent", f"📥 INPUT: Generate ToC for {len(state['clinical_note_text'])} character note")
    logger.log_message("toc_agent", f"🔧 TOOLS: section_detection, json_validation")

    # Invoke agent
    result = agent.invoke({
        "messages": state["messages"] + [input_message]
    })

    # Extract ToC output from agent response
    toc_output = extract_json_from_agent_output(result)

    # Log output preview
    try:
        import json
        toc_data = json.loads(toc_output)
        section_count = len(toc_data.get("sections", []))
        logger.log_message("toc_agent", f"✅ OUTPUT: Generated ToC with {section_count} sections")
        logger.log_message("toc_agent", f"📄 Preview: {json.dumps(toc_data, indent=2)[:300]}...")
    except:
        logger.log_message("toc_agent", f"✅ OUTPUT: {len(toc_output)} characters")

    return {
        "toc_output": toc_output,
        "messages": result["messages"],
    }


def summary_agent_node(state: ClinicalNoteState) -> Dict[str, Any]:
    """
    Summary Subagent: Extract clinical information with citations.

    Args:
        state: Current graph state with clinical note text

    Returns:
        State updates with summary_output (JSON string)
    """
    session_id = state.get("session_id", "unknown")
    logger = AgentLogger(session_id)
    logger.log_message("summary_agent", "Starting clinical summary extraction")

    llm = get_llm()
    prompt = load_agent_prompt("summary-subagent")

    agent = create_react_agent(
        model=llm,
        tools=SKILL_TOOLS["summary_tools"],
        prompt=SystemMessage(content=prompt),
    )

    input_message = HumanMessage(
        content=f"Extract clinical summary from this note:\n\n{state['clinical_note_text']}"
    )

    # Log input
    logger.log_message("summary_agent", f"📥 INPUT: Extract clinical info from {len(state['clinical_note_text'])} character note")
    logger.log_message("summary_agent", f"🔧 TOOLS: contextual_chunking, rag_retrieval, citation_extraction, json_validation")

    result = agent.invoke({
        "messages": state["messages"] + [input_message]
    })

    # Extract Summary output from agent response
    summary_output = extract_json_from_agent_output(result)

    # Log output preview
    try:
        import json
        summary_data = json.loads(summary_output)
        findings_count = len(summary_data.get("findings", []))
        logger.log_message("summary_agent", f"✅ OUTPUT: Extracted {findings_count} clinical findings")
        logger.log_message("summary_agent", f"📄 Preview: {json.dumps(summary_data, indent=2)[:300]}...")
    except:
        logger.log_message("summary_agent", f"✅ OUTPUT: {len(summary_output)} characters")

    return {
        "summary_output": summary_output,
        "messages": result["messages"],
    }


def recommendation_agent_node(state: ClinicalNoteState) -> Dict[str, Any]:
    """
    Recommendation Subagent: Generate treatment plan with confidence scoring.

    Args:
        state: Current graph state with clinical note text and summary

    Returns:
        State updates with plan_output (JSON string)
    """
    session_id = state.get("session_id", "unknown")
    logger = AgentLogger(session_id)
    logger.log_message("recommendation_agent", "Starting treatment recommendation generation")

    llm = get_llm()
    prompt = load_agent_prompt("recommendation-subagent")

    agent = create_react_agent(
        model=llm,
        tools=SKILL_TOOLS["recommendation_tools"],
        prompt=SystemMessage(content=prompt),
    )

    summary_preview = state.get('summary_output', '')[:200] if state.get('summary_output') else 'None'
    input_message = HumanMessage(
        content=f"Generate treatment recommendations based on this summary:\n\n{state.get('summary_output', '')}"
    )

    # Log input
    logger.log_message("recommendation_agent", f"📥 INPUT: Generate recommendations from summary")
    logger.log_message("recommendation_agent", f"📄 Summary preview: {summary_preview}...")
    logger.log_message("recommendation_agent", f"🔧 TOOLS: rag_retrieval, citation_extraction, json_validation")

    result = agent.invoke({
        "messages": state["messages"] + [input_message]
    })

    # Extract Plan output from agent response
    plan_output = extract_json_from_agent_output(result)

    # Log output preview
    try:
        import json
        plan_data = json.loads(plan_output)
        rec_count = len(plan_data.get("recommendations", []))
        logger.log_message("recommendation_agent", f"✅ OUTPUT: Generated {rec_count} treatment recommendations")
        logger.log_message("recommendation_agent", f"📄 Preview: {json.dumps(plan_data, indent=2)[:300]}...")
    except:
        logger.log_message("recommendation_agent", f"✅ OUTPUT: {len(plan_output)} characters")

    return {
        "plan_output": plan_output,
        "messages": result["messages"],
    }


def evaluator_agent_node(state: ClinicalNoteState) -> Dict[str, Any]:
    """
    Evaluator Agent: Validate outputs and provide quality feedback.

    Args:
        state: Current graph state with all three outputs

    Returns:
        State updates with evaluator_feedback, evaluation_status, quality_metrics
    """
    session_id = state.get("session_id", "unknown")
    logger = AgentLogger(session_id)
    iteration = state.get("iteration_count", 0)
    logger.log_message("evaluator_agent", f"Starting quality evaluation (Iteration {iteration})")

    llm = get_llm()
    prompt = load_agent_prompt("evaluator-agent")

    agent = create_react_agent(
        model=llm,
        tools=SKILL_TOOLS["evaluator_tools"],
        prompt=SystemMessage(content=prompt),
    )

    # Prepare outputs for evaluation
    outputs = {
        "toc": state.get("toc_output"),
        "summary": state.get("summary_output"),
        "plan": state.get("plan_output"),
    }

    # Log input
    logger.log_message("evaluator_agent", f"📥 INPUT: Evaluating ToC, Summary, and Recommendations")
    logger.log_message("evaluator_agent", f"🔧 TOOLS: citation_extraction, json_validation")
    logger.log_message("evaluator_agent", f"📊 Iteration: {iteration}/{MAX_ITERATIONS}")

    input_message = HumanMessage(
        content=f"Evaluate these outputs:\n\n{outputs}\n\nIteration: {state.get('iteration_count', 0)}"
    )

    result = agent.invoke({
        "messages": state["messages"] + [input_message]
    })

    # Extract evaluator feedback from agent response
    evaluator_json = extract_json_from_agent_output(result)

    try:
        evaluator_feedback = json.loads(evaluator_json)
    except json.JSONDecodeError:
        # If JSON parsing fails, create a default failure response
        evaluator_feedback = {
            "status": "fail",
            "iteration_number": state["iteration_count"] + 1,
            "issues": ["Failed to parse evaluator output"],
            "suggestions": ["Retry evaluation"],
            "quality_metrics": {
                "citation_coverage": 0.0,
                "hallucination_rate": 1.0,
                "jaccard_overlap": 0.0,
            }
        }

    evaluation_status = evaluator_feedback.get("status", "fail")
    metrics = evaluator_feedback.get("quality_metrics", {})

    # Log detailed evaluation results
    logger.log_message("evaluator_agent", f"✅ EVALUATION: {evaluation_status.upper()}")
    if metrics:
        logger.log_message("evaluator_agent", f"📊 Citation Coverage: {metrics.get('citation_coverage', 0):.2%}")
        logger.log_message("evaluator_agent", f"📊 Hallucination Rate: {metrics.get('hallucination_rate', 0):.2%}")
        logger.log_message("evaluator_agent", f"📊 Jaccard Overlap: {metrics.get('jaccard_overlap', 0):.2%}")

    if evaluation_status == "fail":
        issues = evaluator_feedback.get("issues", [])
        logger.log_message("evaluator_agent", f"⚠️ Issues Found: {', '.join(issues[:3])}")
        suggestions = evaluator_feedback.get("suggestions", [])
        logger.log_message("evaluator_agent", f"💡 Suggestions: {', '.join(suggestions[:3])}")

    return {
        "evaluator_feedback": evaluator_feedback,
        "evaluation_status": evaluation_status,
        "quality_metrics": evaluator_feedback.get("quality_metrics", {}),
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": result["messages"],
    }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_continue_refinement(state: ClinicalNoteState) -> Literal["refine", "complete"]:
    """
    Determine next step after evaluation.

    Args:
        state: Current graph state with evaluation results

    Returns:
        - "complete": Evaluation passed OR max iterations reached, output results
        - "refine": Evaluation failed, refine outputs (if iterations < max)
    """
    if state.get("evaluation_status") == "pass":
        return "complete"

    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        # Max iterations reached - output current results instead of failing
        return "complete"

    return "refine"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_clinical_notes_graph() -> StateGraph:
    """
    Build the LangGraph workflow for clinical notes processing.

    Returns:
        Compiled StateGraph ready for invocation

    Graph Structure:
        START
          ├─> toc_agent
          ├─> summary_agent
          └─> recommendation_agent
                  ↓
              evaluator_agent
                  ↓
              [decision point]
                  ├─> complete (if pass OR iterations >= 2) → END
                  └─> refine (if fail & iterations < 2) → loop back
    """
    # Initialize graph with state schema
    graph = StateGraph(ClinicalNoteState)

    # Add agent nodes
    graph.add_node("toc_agent", toc_agent_node)
    graph.add_node("summary_agent", summary_agent_node)
    graph.add_node("recommendation_agent", recommendation_agent_node)
    graph.add_node("evaluator_agent", evaluator_agent_node)

    # Define edges
    # Start with parallel processing of all three subagents
    # TODO: Implement proper parallel fan-out/fan-in pattern
    # For now, using sequential flow:
    graph.set_entry_point("toc_agent")
    graph.add_edge("toc_agent", "summary_agent")
    graph.add_edge("summary_agent", "recommendation_agent")
    graph.add_edge("recommendation_agent", "evaluator_agent")

    # Conditional edge from evaluator
    graph.add_conditional_edges(
        "evaluator_agent",
        should_continue_refinement,
        {
            "complete": END,  # Evaluation passed OR max iterations reached
            "refine": "toc_agent",  # Loop back to start refinement
        }
    )

    # Compile graph with checkpointing for state persistence
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================================
# MAIN EXECUTION INTERFACE
# ============================================================================

def process_clinical_note(
    note_path: str,
    note_text: str,
    collection_name: str,
    session_id: str = None,
) -> ClinicalNoteState:
    """
    Process a clinical note through the multi-agent graph.

    Args:
        note_path: Path to uploaded clinical note file
        note_text: Full text content of the note
        collection_name: ChromaDB collection name for this session
        session_id: Session ID for observability logging

    Returns:
        Final state with all outputs (toc, summary, plan) and evaluation results

    Example:
        >>> state = process_clinical_note(
        ...     note_path="/tmp/note.txt",
        ...     note_text="Clinical note content...",
        ...     collection_name="session_12345",
        ...     session_id="12345"
        ... )
        >>> print(state["evaluation_status"])  # "pass" or "fail"
        >>> print(state["toc_output"])  # ToC JSON
    """
    # Build graph
    graph = build_clinical_notes_graph()

    # Initialize state
    initial_state = ClinicalNoteState(
        clinical_note_path=note_path,
        clinical_note_text=note_text,
        collection_name=collection_name,
        session_id=session_id or collection_name,
        messages=[],
    )

    # Invoke graph
    config = {"configurable": {"thread_id": collection_name}}
    final_state = graph.invoke(initial_state, config=config)

    return final_state


# ============================================================================
# ENTRYPOINT FOR TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Test the graph with a sample clinical note.
    """
    # Sample note for testing
    test_note = """
    PATIENT: Jane Doe
    AGE: 45
    CHIEF COMPLAINT: Chest pain

    HISTORY OF PRESENT ILLNESS:
    Patient reports intermittent chest pain for 3 days.
    Pain is substernal, pressure-like, radiating to left arm.
    Associated with diaphoresis and nausea.

    ASSESSMENT:
    Acute coronary syndrome - rule out MI

    PLAN:
    1. EKG
    2. Troponin levels
    3. Cardiology consult
    """

    try:
        result = process_clinical_note(
            note_path="/tmp/test_note.txt",
            note_text=test_note,
            collection_name="test_session_001"
        )
        print("Processing complete!")
        print(f"Status: {result.get('evaluation_status')}")
        print(f"Iterations: {result.get('iteration_count')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
