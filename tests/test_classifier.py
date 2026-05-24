"""
Tests for the task classifier.
Run with:  .venv/bin/pytest tests/ -v
"""

import pytest
from byok.core.classifier import TaskClassifier, TaskProfile


@pytest.fixture
def clf():
    return TaskClassifier()


def msg(text: str, role: str = "user") -> list[dict]:
    """Helper: wrap text into messages list."""
    return [{"role": role, "content": text}]


# ── Task type classification ───────────────────────────────────────────────────

class TestTaskTypeClassification:

    def test_coding_function(self, clf):
        profile = clf.classify(msg("Write a Python function that sorts a list by multiple keys"))
        assert profile.task_type == "coding"

    def test_coding_class(self, clf):
        profile = clf.classify(msg("Write a Python class that implements a binary search tree with insert, delete, and search"))
        assert profile.task_type == "coding"

    def test_coding_debug(self, clf):
        profile = clf.classify(msg("Debug this Python code: the function throws a KeyError on line 42"))
        assert profile.task_type == "coding"

    def test_coding_debug_endpoint(self, clf):
        profile = clf.classify(msg("Debug this FastAPI endpoint"))
        assert profile.task_type == "coding"

    def test_coding_sql(self, clf):
        profile = clf.classify(msg("Write a SQL query that joins users and orders tables on user_id and filters by last 30 days"))
        assert profile.task_type == "coding"

    def test_reasoning(self, clf):
        profile = clf.classify(msg("Analyze the pros and cons of microservices vs monolith architecture"))
        assert profile.task_type == "reasoning"

    def test_reasoning_compare(self, clf):
        profile = clf.classify(msg("Compare PostgreSQL and MongoDB and evaluate which is better for my use case"))
        assert profile.task_type == "reasoning"

    def test_math(self, clf):
        profile = clf.classify(msg("Calculate the derivative of f(x) = 3x^2 + 2x - 5"))
        assert profile.task_type == "math"

    def test_math_statistics(self, clf):
        profile = clf.classify(msg("Compute the mean and standard deviation of this dataset: 4, 8, 15, 16, 23, 42"))
        assert profile.task_type == "math"

    def test_writing_email(self, clf):
        profile = clf.classify(msg("Write a professional email declining a job offer politely"))
        assert profile.task_type == "writing"

    def test_writing_essay(self, clf):
        profile = clf.classify(msg("Draft a 500-word essay on the impact of artificial intelligence on healthcare"))
        assert profile.task_type == "writing"

    def test_summarization(self, clf):
        profile = clf.classify(msg("Summarize this article into 3 key bullet points"))
        assert profile.task_type == "summarization"

    def test_summarization_tldr(self, clf):
        profile = clf.classify(msg("tldr of this document please"))
        assert profile.task_type == "summarization"

    def test_simple_chat_greeting(self, clf):
        profile = clf.classify(msg("hi"))
        assert profile.task_type == "simple_chat"

    def test_simple_chat_thanks(self, clf):
        profile = clf.classify(msg("thanks!"))
        assert profile.task_type == "simple_chat"


# ── Tool calling detection ────────────────────────────────────────────────────

class TestToolDetection:

    def test_tools_present_boosts_tool_calling(self, clf):
        tools = [{"type": "function", "function": {"name": "search_web"}}]
        profile = clf.classify(msg("Search for the latest news about AI"), tools=tools)
        assert profile.has_tools is True

    def test_no_tools_when_not_provided(self, clf):
        profile = clf.classify(msg("Write me a poem"))
        assert profile.has_tools is False

    def test_tools_list_respected(self, clf):
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        profile = clf.classify(msg("What's the weather?"), tools=tools)
        assert profile.has_tools is True


# ── Privacy detection ─────────────────────────────────────────────────────────

class TestPrivacyDetection:

    def test_confidential_keyword(self, clf):
        profile = clf.classify(msg("Analyze this confidential employee data"))
        assert profile.privacy_required is True

    def test_private_keyword(self, clf):
        profile = clf.classify(msg("This is private information, keep it local"))
        assert profile.privacy_required is True

    def test_hipaa(self, clf):
        profile = clf.classify(msg("Process this HIPAA-protected patient record"))
        assert profile.privacy_required is True

    def test_normal_task_not_private(self, clf):
        profile = clf.classify(msg("Write a Python function to parse JSON"))
        assert profile.privacy_required is False


# ── Difficulty scoring ────────────────────────────────────────────────────────

class TestDifficultyScoring:

    def test_short_simple_message_is_easy(self, clf):
        profile = clf.classify(msg("hi there"))
        assert profile.difficulty == "easy"

    def test_complex_keyword_boosts_difficulty(self, clf):
        profile = clf.classify(msg("Design a complex distributed system architecture for high availability"))
        assert profile.difficulty in ("medium", "hard")

    def test_multi_part_request_is_harder(self, clf):
        profile = clf.classify(msg(
            "Do the following:\n"
            "1. Write a Python class\n"
            "2. Add unit tests\n"
            "3. Document every method\n"
            "4. Handle edge cases\n"
            "5. Add logging\n"
        ))
        assert profile.difficulty in ("medium", "hard")


# ── Context size ──────────────────────────────────────────────────────────────

class TestContextSize:

    def test_short_message_token_estimate(self, clf):
        profile = clf.classify(msg("hi"))
        assert profile.context_tokens < 20

    def test_long_message_higher_token_estimate(self, clf):
        long_text = "word " * 1000  # 5000 chars ≈ 1250 tokens
        profile = clf.classify(msg(long_text))
        assert profile.context_tokens > 500

    def test_long_input_can_influence_summarization(self, clf):
        long_text = "This is a long article. " * 500
        profile = clf.classify(msg(long_text))
        # Should lean toward summarization due to long-context boost
        assert profile.task_type in ("summarization", "simple_chat")


# ── Confidence ────────────────────────────────────────────────────────────────

class TestConfidence:

    def test_clear_coding_task_high_confidence(self, clf):
        profile = clf.classify(msg("Write a Python function to parse JSON"))
        assert profile.confidence > 0.7

    def test_confidence_bounded(self, clf):
        profile = clf.classify(msg("Write a Python function to parse JSON"))
        assert 0.0 < profile.confidence <= 0.99

    def test_system_prompt_included_in_classification(self, clf):
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Help me fix this bug in my code"},
        ]
        profile = clf.classify(messages)
        assert profile.task_type == "coding"
