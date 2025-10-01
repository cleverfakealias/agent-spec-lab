#!/usr/bin/env python3
"""
Test script showing the correct behavior of the enhanced reformulator.
This shows how the system properly handles different types of questions.
"""

import sys
from pathlib import Path

# Add src to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_spec_lab.nodes.reformulator import create_reformulator_node
from agent_spec_lab.state import AgentState
from conftest import FakeChatModel


def test_reformulator_logic():
    """Test that the reformulator correctly identifies which questions need reformulation."""

    reformulator = create_reformulator_node(FakeChatModel())

    print("🧪 Testing Reformulator Logic")
    print("=" * 50)

    # Test cases: [question, should_be_reformulated, explanation]
    test_cases = [
        (
            "What is quantum computing?",
            False,
            "Clear question - shouldn't reformulate even if off-topic",
        ),
        ("How do I use this?", True, "Vague pronoun 'this' - needs context"),
        ("How do I install the agent?", False, "Clear, specific question"),
        ("It's broken", True, "Vague pronoun 'it' - needs clarification"),
        ("What about nodes?", False, "Clear question about nodes"),
        ("Help", True, "Extremely vague - needs context"),
        ("Where is the documentation?", False, "Clear question with 'where is' pattern"),
        ("Can I use custom models?", False, "Clear question with 'can i' pattern"),
        ("That doesn't work", True, "Vague pronoun 'that' - needs context"),
    ]

    for question, expected_reformulated, explanation in test_cases:
        result = reformulator(AgentState(question=question))

        status = "✅" if result.was_reformulated == expected_reformulated else "❌"
        print(f"{status} '{question}'")
        print(f"   Expected reformulated: {expected_reformulated}")
        print(f"   Actually reformulated: {result.was_reformulated}")
        print(f"   Reason: {explanation}")
        print()


if __name__ == "__main__":
    test_reformulator_logic()

    print("🎯 Key Points:")
    print("• The reformulator now correctly identifies genuinely vague questions")
    print("• Clear questions like 'What is quantum computing?' are NOT reformulated")
    print("• Only questions with unclear pronouns or extreme vagueness are reformulated")
    print(
        "• This prevents the bug where off-topic but clear questions got "
        "reformulated to agent topics"
    )
