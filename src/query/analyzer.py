"""
analyzer.py
-----------
Turns a raw NL query into structured signal: what is the user trying to do
(intent), what technical concepts are they mentioning (entities), and what
identifier-like tokens can we pull out or guess (identifiers).

This is rule-based on purpose — no model call, so it's instant and has zero
failure modes. It's the foundation the other modules build on.

FIX (see demo screenshot bug): the PascalCase check used to match ANY word
starting with a capital letter followed by a lowercase letter -- e.g. "How"
at the start of a sentence -- and wrongly treated it as a real identifier
fragment. Real PascalCase identifiers (like "SessionCreated") have a SECOND
capital letter later in the word; plain capitalized English words don't.
The fix requires that second capital before treating a token as PascalCase.
"""

import re
from dataclasses import dataclass, field
from typing import List


# --- Intent detection --------------------------------------------------
# Each intent maps to a list of trigger phrases. First match wins, checked
# in order of specificity (most specific intents first).
INTENT_PATTERNS = {
    "debug_error": [
        r"\berror\b", r"\bexception\b", r"\bfails?\b", r"\bcrash(es|ing)?\b",
        r"\bbug\b", r"\btraceback\b", r"why (is|does|isn't|doesn't)",
    ],
    "locate_definition": [
        r"\bwhere is\b", r"\bwhere (is|are) .* defined\b", r"\bfind the\b",
        r"\blocate\b", r"\bwhich file\b", r"\bwhich function\b",
    ],
    "usage_example": [
        r"\bexample of\b", r"\bhow (do|to) (i |you )?use\b", r"\bcall(ing)?\b.*\bfunction\b",
    ],
    "how_to": [
        r"^how (is|are|does|do)\b", r"\bhow to\b",
    ],
    "what_is": [
        r"^what (is|are|does)\b", r"\bpurpose of\b", r"\bwhat does .* do\b",
    ],
}

# --- Entity vocabulary ---------------------------------------------------
# Small hand-built list of domain-technical nouns worth flagging. Extend
# this as you see real query patterns during testing.
TECHNICAL_ENTITIES = {
    "input", "output", "validation", "authentication", "authorization",
    "token", "session", "database", "query", "request", "response",
    "config", "configuration", "cache", "queue", "thread", "lock",
    "exception", "error", "log", "logging", "user", "password",
    "encryption", "decryption", "hash", "index", "connection", "socket",
    "api", "endpoint", "middleware", "serializer", "parser", "schema",
}

# A real PascalCase identifier has a lowercase run followed by ANOTHER
# capital letter (e.g. "SessionCreated" -> Session + Created). A plain
# capitalized English word like "How" or "The" does not. This distinguishes
# the two so ordinary sentence-starting words never get treated as code
# identifiers.
_PASCAL_CASE = re.compile(r"^[A-Z][a-z]+[A-Z][a-zA-Z]*$")
_CAMEL_CASE = re.compile(r"[a-z][A-Z]")


@dataclass
class QueryAnalysis:
    raw_query: str
    intent: str
    entities: List[str] = field(default_factory=list)
    identifiers: List[str] = field(default_factory=list)


class QueryAnalyzer:
    def detect_intent(self, query: str) -> str:
        q = query.lower().strip()
        for intent, patterns in INTENT_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, q):
                    return intent
        return "other"

    def extract_entities(self, query: str) -> List[str]:
        q = query.lower()
        tokens = re.findall(r"[a-zA-Z]+", q)
        found = [t for t in tokens if t in TECHNICAL_ENTITIES]
        # de-dupe, preserve order
        seen = set()
        out = []
        for f in found:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    def extract_identifiers(self, query: str) -> List[str]:
        """
        Pull out anything that already looks like an identifier
        (snake_case, camelCase, real PascalCase, or dotted.path), plus
        generate plausible identifier forms from multi-word technical
        phrases found in the query (e.g. "user input" -> "user_input",
        "userInput").
        """
        identifiers = set()

        # Already identifier-shaped tokens in the raw text
        for tok in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query):
            if "_" in tok or _CAMEL_CASE.search(tok) or _PASCAL_CASE.match(tok):
                identifiers.add(tok)

        # Guess identifier forms from adjacent *content* words (skip stopwords)
        STOPWORDS = {"the", "a", "an", "is", "are", "on", "in", "to", "of",
                     "before", "after", "does", "do", "why", "how", "where"}
        words = [w for w in re.findall(r"[a-zA-Z]+", query.lower()) if w not in STOPWORDS]
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 in TECHNICAL_ENTITIES or w2 in TECHNICAL_ENTITIES:
                identifiers.add(f"{w1}_{w2}")
                identifiers.add(w1 + w2.capitalize())

        return sorted(identifiers)

    def analyze(self, query: str) -> QueryAnalysis:
        return QueryAnalysis(
            raw_query=query,
            intent=self.detect_intent(query),
            entities=self.extract_entities(query),
            identifiers=self.extract_identifiers(query),
        )


if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    test_queries = [
        "How is a session created?",
        "how is the input validated before saving to the database",
        "where is the user authentication token generated",
        "why does the connection fail on retry",
        "example of using the cache decorator",
    ]
    for q in test_queries:
        result = analyzer.analyze(q)
        print(f"\nQuery: {q}")
        print(f"  intent:      {result.intent}")
        print(f"  entities:    {result.entities}")
        print(f"  identifiers: {result.identifiers}")
