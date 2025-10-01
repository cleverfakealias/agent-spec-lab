"""
Enhanced uncertainty handling strategies for RAG systems.
Based on current best practices and research in the field.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent_spec_lab.state import AgentState


class UncertaintyReason(str, Enum):
    """Reasons for uncertainty in RAG responses."""

    OUT_OF_SCOPE = "out_of_scope"  # Question not related to knowledge domain
    INSUFFICIENT_CONTEXT = "insufficient"  # Not enough information in retrieval
    CONFLICTING_SOURCES = "conflicting"  # Retrieved info contradicts itself
    AMBIGUOUS_QUERY = "ambiguous"  # Question is unclear/has multiple interpretations
    LOW_RETRIEVAL_SCORE = "low_retrieval"  # Retrieved documents have low similarity
    PARTIAL_MATCH = "partial"  # Only partial answer available


class UncertaintyStrategy(str, Enum):
    """Strategies for handling different types of uncertainty."""

    EXPLICIT_REFUSAL = "explicit_refusal"  # "I don't know" response
    CLARIFICATION = "clarification"  # Ask for more details
    PARTIAL_ANSWER = "partial_answer"  # Provide what's available + acknowledge gaps
    SCOPE_REDIRECT = "scope_redirect"  # Redirect to appropriate resources
    SUGGEST_ALTERNATIVES = "suggest_alternatives"  # Offer related topics


_UNCERTAINTY_ANALYSIS_PROMPT = """
Analyze this Q&A scenario for uncertainty and appropriateness:

Question: {question}
Retrieved Context: {context}
Confidence Score: {confidence_score}

Determine:
1. Is this question within the scope of an agent/FAQ system about Python frameworks?
2. Is there sufficient context to answer confidently?
3. What type of uncertainty exists (if any)?

Respond in this format:
Scope: [in_scope/out_of_scope]
Sufficiency: [sufficient/insufficient/partial]
Uncertainty Type: [none/out_of_scope/insufficient/conflicting/ambiguous/low_retrieval/partial]
Recommended Strategy: [explicit_refusal/clarification/partial_answer/
                      scope_redirect/suggest_alternatives]
Explanation: [brief explanation of your assessment]
""".strip()


def create_uncertainty_handler_node(llm: BaseChatModel) -> Callable[[AgentState], AgentState]:
    """Create a node that intelligently handles uncertainty and out-of-scope questions."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at analyzing uncertainty in Q&A systems."),
            ("human", _UNCERTAINTY_ANALYSIS_PROMPT),
        ]
    )

    def handle_uncertainty(state: AgentState) -> AgentState:
        context = "\n\n".join(state.context) if state.context else "No context retrieved."
        confidence = state.confidence_score or 5

        # Get uncertainty analysis
        messages = prompt.format_messages(
            question=state.question, context=context, confidence_score=confidence
        )

        response = llm.invoke(messages)
        analysis = getattr(response, "content", str(response))

        # Parse the analysis (simplified parsing for demo)
        uncertainty_info = _parse_uncertainty_analysis(analysis)

        # Generate appropriate response based on uncertainty type
        enhanced_answer = _generate_uncertainty_response(
            state.question,
            context,
            uncertainty_info,
            state.answer or "No initial answer generated.",
        )

        return state.model_copy(
            update={
                "answer": enhanced_answer,
                "uncertainty_type": uncertainty_info.get("type", "unknown"),
                "response_strategy": uncertainty_info.get("strategy", "explicit_refusal"),
                "response_explanation": uncertainty_info.get("explanation", ""),
            }
        )

    return handle_uncertainty


def _parse_uncertainty_analysis(analysis: str) -> dict[str, str]:
    """Parse the LLM's uncertainty analysis."""
    info = {}
    lines = analysis.split("\n")

    for line in lines:
        if line.startswith("Scope:"):
            info["scope"] = line.split(":", 1)[1].strip()
        elif line.startswith("Sufficiency:"):
            info["sufficiency"] = line.split(":", 1)[1].strip()
        elif line.startswith("Uncertainty Type:"):
            info["type"] = line.split(":", 1)[1].strip()
        elif line.startswith("Recommended Strategy:"):
            info["strategy"] = line.split(":", 1)[1].strip()
        elif line.startswith("Explanation:"):
            info["explanation"] = line.split(":", 1)[1].strip()

    return info


def _generate_uncertainty_response(
    question: str, context: str, uncertainty_info: dict[str, str], original_answer: str
) -> str:
    """Generate appropriate response based on uncertainty analysis."""

    strategy = uncertainty_info.get("strategy", "explicit_refusal")
    uncertainty_type = uncertainty_info.get("type", "unknown")

    if strategy == "explicit_refusal":
        return _create_explicit_refusal(question, uncertainty_type)
    elif strategy == "scope_redirect":
        return _create_scope_redirect(question, uncertainty_type)
    elif strategy == "partial_answer":
        return _create_partial_answer(question, context, original_answer)
    elif strategy == "clarification":
        return _create_clarification_request(question)
    elif strategy == "suggest_alternatives":
        return _create_alternative_suggestions(question)
    else:
        return _create_explicit_refusal(question, uncertainty_type)


def _create_explicit_refusal(question: str, uncertainty_type: str) -> str:
    """Create a transparent 'I don't know' response."""

    base_response = "I don't have enough reliable information to answer your question confidently."

    if uncertainty_type == "out_of_scope":
        return f"""{base_response}

Your question about "{question}" appears to be outside the scope of this 
agent-spec-lab FAQ system, which focuses on:
• Installing and setting up the Python agent framework
• Understanding LangGraph nodes and workflows  
• Troubleshooting common framework issues
• Learning about project features and capabilities

For questions about other topics, I'd recommend consulting specialized resources 
or search engines."""

    elif uncertainty_type == "insufficient":
        return f"""{base_response}

The available documentation doesn't contain sufficient information to answer "{question}" properly. 

To get accurate information, you might want to:
• Check the project's GitHub repository for more detailed documentation
• Look for examples in the codebase or test files
• Ask the maintainers directly in GitHub issues
• Consult the official project documentation if available"""

    else:
        return f"""{base_response}

While I found some potentially relevant information, I cannot provide a confident 
answer to "{question}" based on the current knowledge base.

For the most accurate information, please refer to the official documentation 
or contact the project maintainers."""


def _create_scope_redirect(question: str, uncertainty_type: str) -> str:
    """Redirect to appropriate resources for out-of-scope questions."""

    return """Your question appears to be outside my area of expertise. I'm designed to help 
specifically with the agent-spec-lab Python framework.

For your question, you might find better answers at:
• General Python documentation (docs.python.org) for Python language questions
• Stack Overflow for programming questions
• Relevant specialized communities or documentation for your specific topic
• Search engines for general knowledge questions

Is there anything about the agent-spec-lab framework specifically that I can 
help you with instead?"""


def _create_partial_answer(question: str, context: str, original_answer: str) -> str:
    """Provide partial information with clear limitations."""

    return f"""I can provide some relevant information about "{question}", but my 
knowledge is limited:

{original_answer}

⚠️ **Important limitations:**
• This information may be incomplete
• The documentation might not cover all aspects of your question
• For comprehensive guidance, please consult additional resources

**Recommended next steps:**
• Check the project's GitHub repository for complete documentation
• Look for related examples in the codebase
• Consider asking the community or maintainers for complete guidance"""


def _create_clarification_request(question: str) -> str:
    """Ask for clarification on ambiguous questions."""

    return f"""Your question "{question}" could be interpreted in several ways. 
To provide the most helpful answer, could you clarify:

• What specific aspect are you most interested in?
• Are you looking for installation steps, usage examples, or troubleshooting help?
• What have you already tried, if anything?
• What's your current setup or context?

The more specific details you can provide, the better I can assist you with 
the agent-spec-lab framework."""


def _create_alternative_suggestions(question: str) -> str:
    """Suggest related topics when exact match isn't available."""

    return f"""I don't have specific information about "{question}", but here are 
some related topics I can help with:

**Installation & Setup:**
• How to install the agent-spec-lab project
• Setting up your development environment
• Configuration requirements

**Framework Usage:**
• Creating and using LangGraph nodes
• Building agent workflows
• Understanding the project architecture

**Troubleshooting:**
• Common installation issues
• Debugging agent workflows
• Performance considerations

Would any of these topics be helpful, or is there a specific aspect of the 
framework you'd like to explore?"""


# Confidence threshold configurations
CONFIDENCE_THRESHOLDS = {
    "high_confidence": 8,  # Answer directly
    "medium_confidence": 6,  # Answer with caveats
    "low_confidence": 4,  # Use uncertainty handling
    "very_low_confidence": 2,  # Explicit refusal
}


def should_handle_uncertainty(state: AgentState) -> bool:
    """Determine if uncertainty handling is needed."""

    confidence = state.confidence_score or 5

    # Always handle if confidence is very low
    if confidence <= CONFIDENCE_THRESHOLDS["very_low_confidence"]:
        return True

    # Handle if no context was retrieved (indicates no relevant info found)
    if not state.context or all(not ctx.strip() for ctx in state.context):
        return True

    # Handle if question is empty or extremely vague
    if not state.question or len(state.question.strip()) < 3:
        return True

    # Handle if low confidence (even with some context)
    if confidence <= CONFIDENCE_THRESHOLDS["low_confidence"]:
        return True

    # Handle if question seems off-topic (basic keyword check)
    off_topic_keywords = [
        "quantum",
        "physics",
        "chemistry",
        "biology",
        "cooking",
        "sports",
        "weather",
        "politics",
        "medicine",
        "law",
        "finance",
        "astronomy",
    ]
    if any(keyword in state.question.lower() for keyword in off_topic_keywords):
        return True

    return False


__all__ = [
    "create_uncertainty_handler_node",
    "UncertaintyReason",
    "UncertaintyStrategy",
    "should_handle_uncertainty",
    "CONFIDENCE_THRESHOLDS",
]
