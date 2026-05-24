"""
classifier.py — Task Classifier

Reads the incoming chat messages and figures out:
  - What KIND of task is this? (coding, reasoning, summarization, etc.)
  - How HARD is it? (easy, medium, hard)
  - How BIG is the context? (estimated token count)
  - Does it use tools?
  - Does it require privacy (local-only)?

This is the brain of BYOK. The better it classifies, the better it routes.
V1 uses keyword pattern matching — no ML required, fully transparent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Keyword patterns per task type
#  Each entry: (regex_pattern, score_weight)
#  A task type wins if it accumulates the highest total score.
# ─────────────────────────────────────────────────────────────────────────────

TASK_SIGNALS: dict[str, list[tuple[str, float]]] = {
    "coding": [
        (r"\b(write|create|build|implement)\s+a?\s*(function|class|script|program|api|endpoint|module|component|service)\b", 3.0),
        (r"\b(debug|fix|refactor|optimize|review|test)\s+(this\s+)?(code|function|script|bug|error|issue|api|endpoint|service|component|module)\b", 3.0),
        (r"\b(fastapi|django|flask|express|next\.js|nextjs|api\s+route|endpoint)\b", 2.0),
        (r"\b(python|javascript|typescript|rust|go|java|sql|bash|html|css|react|node|c\+\+|kotlin|swift)\b", 2.0),
        (r"```[\w]*",                    2.0),   # code block in message
        (r"\b(def |class |import |from |const |let |var |func |fn )\b", 2.0),
        (r"\b(algorithm|data\s+structure|recursion|iteration|complexity|big.?o)\b", 2.0),
        (r"\b(unit\s+test|integration\s+test|mock|pytest|jest|unittest)\b", 2.0),
        (r"\b(sql|query|select|insert|update|delete|join|database|schema)\b", 1.5),
        (r"\b(docker|kubernetes|ci/?cd|pipeline|deploy|dockerfile)\b", 1.5),
    ],

    "reasoning": [
        (r"\b(analyze|analyse|explain\s+why|reason\s+through|think\s+through|walk\s+me\s+through)\b", 3.0),
        (r"\b(compare|contrast|evaluate|assess|weigh|critique)\b", 2.5),
        (r"\b(pros?\s+and\s+cons?|trade-?offs?|advantages?|disadvantages?)\b", 2.5),
        (r"\b(should\s+i|which\s+is\s+better|what\s+would\s+you\s+recommend|best\s+(approach|option|way))\b", 2.0),
        (r"\b(implications?|consequences?|impact|effect|result)\s+of\b", 2.0),
        (r"\b(strategy|decision|framework|methodology|approach)\b", 1.5),
        (r"\b(given\s+that|assuming\s+that|if\s+.{5,30}\s+then)\b", 1.5),
    ],

    "math": [
        (r"\b(calculate|compute|solve|simplify|prove|derive)\b", 3.0),
        (r"\b(equation|formula|integral|derivative|matrix|vector|tensor)\b", 3.0),
        (r"\b(statistics|probability|algebra|geometry|calculus|linear\s+algebra)\b", 2.5),
        (r"\d+\s*[\+\-\*\/\^]\s*\d+",   2.0),   # arithmetic expression
        (r"\b(mean|median|mode|variance|standard\s+deviation)\b", 2.0),
        (r"\\frac|\\sum|\\int|\\sqrt",   2.0),   # LaTeX math
        (r"\b(optimize|minimize|maximize|gradient|loss\s+function)\b", 1.5),
    ],

    "writing": [
        # Match "write/draft/compose X" even with words in between (e.g. "draft a 500-word essay")
        (r"\b(write|draft|compose|create)\b.{0,30}?\b(email|letter|essay|blog\s+post|article|story|report|cover\s+letter|proposal)\b", 3.0),
        (r"\b(proofread|edit|improve|rewrite|rephrase|paraphrase|polish)\b", 2.5),
        (r"\b(tone|style|voice|formal|informal|professional|persuasive|creative)\b", 2.0),
        (r"\b(introduction|conclusion|paragraph|section|chapter|outline)\b", 1.5),
        (r"\b(grammar|spelling|punctuation|word\s+choice|clarity|concise)\b", 1.5),
    ],

    "summarization": [
        (r"\b(summarize|summarise|sum\s+up|condense|compress|shorten)\b", 3.0),
        (r"\btldr\b|\btl;dr\b",          3.0),
        (r"\b(key\s+points?|main\s+points?|highlights?|takeaways?|gist|essence)\b", 2.5),
        (r"\b(brief|overview|recap|synopsis|abstract|digest)\b", 2.0),
        (r"\b(what\s+(are\s+the\s+)?(main|key|important)\s+(points?|ideas?|findings?))\b", 2.0),
    ],

    "tool_calling": [
        # These are detected from the presence of tools in the request,
        # but we also look for keywords as hints.
        (r"\b(search\s+(the\s+)?(web|internet|online)|look\s+up|find\s+information)\b", 2.0),
        (r"\b(fetch|retrieve|get|download|scrape)\s+(data|information|content|file)\b", 2.0),
        (r"\b(call|invoke|trigger|execute|run)\s+a?\s*(function|tool|api|endpoint)\b", 2.0),
        (r"\b(send\s+(an?\s+)?(email|message|notification)|post\s+to)\b", 1.5),
    ],

    "simple_chat": [
        (r"^(hi|hello|hey|good\s+(morning|afternoon|evening))[!,.]?\s*$", 3.0),
        (r"^(thanks|thank\s+you|thx|ty)[!,.]?\s*$", 3.0),
        (r"\b(what\s+is|what's|who\s+is|when\s+did|where\s+is|how\s+do\s+you)\b", 1.0),
        (r"\b(tell\s+me\s+about|explain\s+to\s+me\s+what)\b", 0.5),
    ],
}

# Keywords that signal the task requires a local/private model
PRIVACY_SIGNALS = [
    r"\b(private|confidential|secret|sensitive|internal|proprietary|personal\s+data)\b",
    r"\b(do\s+not\s+(send|share|upload)|keep\s+(this\s+)?local|offline)\b",
    r"\b(HIPAA|GDPR|PII|PHI)\b",
]

# Keywords that suggest the task is harder than average
DIFFICULTY_BOOST_SIGNALS = [
    r"\b(complex|advanced|sophisticated|non-?trivial|nuanced|in-?depth)\b",
    r"\b(multi-?step|end-?to-?end|full\s+system|architecture|design)\b",
    r"\b(research|comprehensive|exhaustive|detailed\s+analysis)\b",
]


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskProfile:
    """
    What BYOK knows about the incoming task.
    This is passed to the Router, which uses it to pick a model.
    """
    task_type: str              # primary task type
    secondary_types: list[str]  # other types that also scored high
    difficulty: str             # easy | medium | hard
    context_tokens: int         # rough estimate of input size
    has_tools: bool             # tools[] was present in the request
    privacy_required: bool      # must use a local model
    confidence: float           # 0.0–1.0 (how sure we are about task_type)

    def __str__(self) -> str:
        tools_note = " [tools]" if self.has_tools else ""
        private_note = " [PRIVATE]" if self.privacy_required else ""
        return (
            f"{self.task_type} ({self.difficulty})"
            f"{tools_note}{private_note}"
            f" ~{self.context_tokens} tokens"
            f" [{self.confidence:.0%} confident]"
        )


class TaskClassifier:
    """
    Reads the incoming messages and returns a TaskProfile.

    It works by:
    1. Joining all message content into a single text blob
    2. Running each task type's patterns against it
    3. Picking the type with the highest accumulated score
    4. Estimating difficulty from message complexity
    5. Checking for privacy keywords
    """

    # Rough token estimate: 1 token ≈ 4 characters
    CHARS_PER_TOKEN = 4

    def classify(
        self,
        messages: list[dict],
        tools: Optional[list] = None,
    ) -> TaskProfile:
        """
        Main entry point. Pass the messages list from the OpenAI-format request.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            tools: The tools[] array from the request (if any)

        Returns:
            A TaskProfile describing what kind of task this is
        """
        # Combine all message content into one string for pattern matching
        full_text = self._extract_text(messages)
        context_tokens = len(full_text) // self.CHARS_PER_TOKEN

        # Score each task type
        scores: dict[str, float] = {}
        for task_type, patterns in TASK_SIGNALS.items():
            scores[task_type] = self._score_task_type(full_text, patterns)

        # If tools are present, boost tool_calling score significantly
        if tools:
            scores["tool_calling"] = scores.get("tool_calling", 0.0) + 5.0

        # Long inputs lean toward summarization
        if context_tokens > 8000:
            scores["summarization"] = scores.get("summarization", 0.0) + 3.0

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # Fall back to simple_chat if nothing scored
        if best_score < 1.0:
            best_type = "simple_chat"
            best_score = 1.0

        # Confidence: how much did the winner beat the runner-up?
        total = best_score + second_score
        confidence = best_score / total if total > 0 else 0.5
        confidence = min(confidence, 0.99)

        # Secondary types (scored at least 30% of the winner)
        threshold = best_score * 0.3
        secondary = [t for t, s in ranked[1:] if s >= threshold and t != best_type]

        return TaskProfile(
            task_type=best_type,
            secondary_types=secondary[:2],
            difficulty=self._estimate_difficulty(full_text, context_tokens),
            context_tokens=context_tokens,
            has_tools=bool(tools),
            privacy_required=self._check_privacy(full_text),
            confidence=confidence,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_text(self, messages: list[dict]) -> str:
        """Pull all text content from the messages list."""
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # Handle multi-part messages (text + images, etc.)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part["text"])
        return "\n".join(parts)

    def _score_task_type(
        self, text: str, patterns: list[tuple[str, float]]
    ) -> float:
        """Sum the weights of all patterns that match the text."""
        score = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                score += weight
        return score

    def _estimate_difficulty(self, text: str, context_tokens: int) -> str:
        """
        Rough difficulty estimate based on:
        - Message length
        - Presence of complexity keywords
        - Number of questions or requirements
        """
        score = 0

        # Length
        if context_tokens > 3000:
            score += 2
        elif context_tokens > 800:
            score += 1

        # Count ALL matching complexity signals (removed early break)
        for pattern in DIFFICULTY_BOOST_SIGNALS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1

        # Multiple questions or bullet points (compound task)
        question_count = text.count("?")
        bullet_count = len(re.findall(r"^\s*[-*•]\s", text, re.MULTILINE))
        if question_count > 2 or bullet_count > 4:
            score += 1

        # Numbered list of 3+ requirements = multi-step = harder
        numbered = len(re.findall(r"^\s*\d+[\.\)]\s", text, re.MULTILINE))
        if numbered >= 3:
            score += 1

        if score >= 4:
            return "hard"
        elif score >= 1:   # lowered from 2 → 1: any signal = at least medium
            return "medium"
        return "easy"

    def _check_privacy(self, text: str) -> bool:
        """Return True if any privacy keywords are found."""
        for pattern in PRIVACY_SIGNALS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
