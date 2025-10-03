"""Enhanced fallback node with responsible AI handling."""

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agent_spec_lab.state import AgentState
from agent_spec_lab.tools.logging import StructuredLogger, trace_node


def create_fallback_node(llm: BaseChatModel) -> Callable[[AgentState], AgentState]:
    """Create a responsible fallback node that provides helpful guidance when unable to answer."""

    logger = StructuredLogger("fallback")

    @trace_node("fallback")
    def handle_fallback(state: AgentState) -> AgentState:
        # Analyze the question to provide contextually appropriate fallback
        question = state.question.lower() if state.question else ""
        confidence_score = state.confidence_score or 0

        logger.info(
            "Executing fallback handler",
            state=state,
            confidence_score=confidence_score,
            question_length=len(question),
        )

        # Generate different fallback responses based on the situation
        harmful_words = ["hack", "crack", "exploit", "attack", "malware"]
        if question and any(harmful_word in question for harmful_word in harmful_words):
            logger.warning(
                "Detected potentially harmful question",
                state=state,
                detected_words=[word for word in harmful_words if word in question],
            )
            # Handle potentially harmful requests
            answer = """I can't provide assistance with activities that could be harmful
or illegal. 

If you're interested in cybersecurity or ethical hacking, I'd recommend:
• Learning through legitimate educational platforms
• Exploring capture-the-flag (CTF) competitions
• Studying ethical hacking certifications

Is there something about the agent-spec-lab framework I can help you with instead?"""

        elif not question or len(question.strip()) < 3:
            # Handle empty or very vague questions
            answer = """I'd be happy to help! However, your question seems incomplete. 

I'm designed to assist with the agent-spec-lab Python framework, including:
• Installation and setup
• Understanding LangGraph workflows
• Troubleshooting common issues
• Learning about project features

Could you please provide more details about what you'd like to know?"""

        elif question and any(
            off_topic in question
            for off_topic in ["weather", "cooking", "sports", "politics", "medicine"]
        ):
            # Handle clearly off-topic questions
            answer = f"""I don't have information about that topic. I'm specialized in helping
with the agent-spec-lab Python framework.

For questions about "{state.question}", you might want to try:
• General search engines for broad topics
• Specialized forums or communities
• Official documentation for specific tools or services

Is there anything about agent-spec-lab, LangGraph, or Python development that I can
help you with?"""

        elif confidence_score <= 3:
            # Handle low confidence scenarios
            answer = """I don't have reliable information to answer your question confidently.

This could be because:
• Your question might be outside my knowledge area (agent-spec-lab framework)
• The available documentation doesn't cover this specific topic
• Your question might need clarification

For the best results, you might want to:
• Check the project's GitHub repository for detailed documentation
• Browse through the example code and test files
• Ask the maintainers directly in GitHub issues

Is there a different way I can help you with agent-spec-lab?"""

        else:
            # General fallback for medium confidence scenarios
            answer = """I found some relevant information, but I'm not confident enough to
provide a complete answer.

To get accurate information about your question, I recommend:
• Consulting the project's official documentation
• Looking at example implementations in the codebase
• Checking recent discussions in GitHub issues

Would you like me to help you find specific resources, or do you have a related 
question I might be able to answer more confidently?"""

        fallback_type = (
            "harmful_content"
            if any(word in question for word in harmful_words)
            else "low_confidence"
        )

        logger.info(
            "Fallback response generated",
            state=state,
            fallback_type=fallback_type,
            answer_length=len(answer),
            confidence_score=confidence_score,
        )

        return state.model_copy(
            update={
                "answer": answer,
                "is_fallback_response": True,
                "response_strategy": "responsible_fallback",
                "response_explanation": (
                    f"Fallback triggered due to confidence level {confidence_score} or "
                    "question type analysis"
                ),
            }
        )

    return handle_fallback
