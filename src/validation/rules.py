from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Data structures

@dataclass(frozen=True)
class Issue:
    code: str
    severity: str  # "INFO" | "WARN" | "ERROR"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    passed: bool
    issues: List[Issue] = field(default_factory=list)

    # Optional: pipeline can choose to use these
    corrected_answer: Optional[str] = None
    blocked: bool = False

    # Useful UI/meta
    confidence: str = "unknown"  # "high" | "medium" | "low" | "unknown"


# Config 

IDK_PHRASE = "I don't know based on the provided documents."

CURRENCY_RE = re.compile(
    r"""
    (?:
      €\s?\d[\d,.\s]* |
      \d[\d,.\s]*\s?€ |
      \bEUR\b\s?\d[\d,.\s]* |
      \b\d[\d,.\s]*\s?\bEUR\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

TEST_SCORE_RE = re.compile(
    r"\b(TOEFL|IELTS|Duolingo)\b.*?\b(\d{2,3}(\.\d)?)\b", re.IGNORECASE
)

PHONE_RE = re.compile(r"\+\d[\d\s()/-]{6,}")

# Any explicit numeric claim we care about (fees, CP/credits, scores)
NUMBER_RE = re.compile(r"\b\d{1,3}(?:[,\.\s]\d{3})*(?:\.\d+)?\b")


def validate_response(
    *,
    query: str,
    response: Dict[str, Any],
    expected_source: Optional[str] = None,  
) -> ValidationResult:
    """
    Validate the output of ask_kb().

    Parameters
    ----------
    query:
        The user question.
    response:
        The dict returned by ask_kb(): {"mode","answer","citations","hits",...}
    expected_source:
        If the UI requested a specific source, pass it here. If None -> don't enforce.

    Returns
    -------
    ValidationResult
    """
    issues: List[Issue] = []

    mode = (response.get("mode") or "").strip().lower()
    answer = (response.get("answer") or "").strip()
    citations = response.get("citations") or []
    hits = response.get("hits") or []

    # Run rules
    issues.extend(_rule_basic_schema(mode, answer, citations))
    issues.extend(_rule_source_scope(citations, expected_source))
    issues.extend(_rule_no_evidence_means_idk(answer, citations, hits))
    issues.extend(_rule_numbers_must_be_supported(answer, citations))
    issues.extend(_rule_idk_should_not_have_strong_evidence(answer, citations))
    issues.extend(_rule_mode_sanity(query, mode))

    passed = not any(i.severity == "ERROR" for i in issues)

    # Simple confidence heuristic
    confidence = _confidence_from_issues(issues, citations)

    return ValidationResult(
        passed=passed,
        issues=issues,
        corrected_answer=None,
        blocked=False,
        confidence=confidence,
    )


# Rules

def _rule_basic_schema(mode: str, answer: str, citations: Sequence[Dict[str, Any]]) -> List[Issue]:
    out: List[Issue] = []
    if mode not in ("answer", "list"):
        out.append(
            Issue(
                code="SCHEMA_MODE_INVALID",
                severity="ERROR",
                message=f"Invalid mode '{mode}'. Expected 'answer' or 'list'.",
                details={"mode": mode},
            )
        )

    if not answer:
        out.append(
            Issue(
                code="SCHEMA_EMPTY_ANSWER",
                severity="ERROR",
                message="Answer is empty.",
            )
        )

    if citations is None:
        out.append(
            Issue(
                code="SCHEMA_CITATIONS_NONE",
                severity="WARN",
                message="Citations is None; expected a list (possibly empty).",
            )
        )

    return out


def _rule_source_scope(
    citations: Sequence[Dict[str, Any]],
    expected_source: Optional[str],
) -> List[Issue]:
    """
    If UI requests a specific source, ensure citations come only from that source.
    """
    if not expected_source:
        return []

    expected = expected_source.strip().lower()
    if expected not in ("pdf", "web", "excel"):
        return [
            Issue(
                code="SOURCE_SCOPE_INVALID_EXPECTED",
                severity="WARN",
                message=f"expected_source '{expected_source}' is not one of pdf/web/excel; skipping enforcement.",
            )
        ]

    bad = []
    for c in citations:
        src = (c.get("source") or "").strip().lower()
        if src and src != expected:
            bad.append(src)

    if bad:
        return [
            Issue(
                code="SOURCE_SCOPE_MISMATCH",
                severity="WARN",
                message=f"Citations include sources outside expected '{expected}'.",
                details={"expected_source": expected, "found_sources": sorted(set(bad))},
            )
        ]
    return []


def _rule_no_evidence_means_idk(
    answer: str,
    citations: Sequence[Dict[str, Any]],
    hits: Sequence[Dict[str, Any]],
) -> List[Issue]:
    """
    If there's effectively no evidence retrieved, the answer should be IDK.
    This is your key safety rule.
    """
    out: List[Issue] = []

    # "No evidence" heuristic
    has_citations = len(citations) > 0

    max_hit_score = 0.0
    for h in hits:
        try:
            max_hit_score = max(max_hit_score, float(h.get("score", 0.0)))
        except Exception:
            pass

    no_evidence = (not has_citations) and (len(hits) == 0 or max_hit_score < 0.20)

    if no_evidence and IDK_PHRASE.lower() not in answer.lower():
        out.append(
            Issue(
                code="NO_EVIDENCE_NOT_IDK",
                severity="ERROR",
                message="No evidence found, but answer is not an 'I don't know' response.",
                details={"max_hit_score": max_hit_score, "citations": len(citations), "hits": len(hits)},
            )
        )

    return out


def _rule_numbers_must_be_supported(answer: str, citations: Sequence[Dict[str, Any]]) -> List[Issue]:
    """
    If the answer contains numeric claims (fees, TOEFL/IELTS, etc.),
    require the citations to include those numbers OR at least include relevant numeric evidence.

    This prevents hallucinated fees/scores.
    """
    out: List[Issue] = []

    # Detect "high-risk" numeric claims
    mentions_currency = bool(CURRENCY_RE.search(answer))
    mentions_test_scores = bool(TEST_SCORE_RE.search(answer))
    mentions_phone = bool(PHONE_RE.search(answer))

       
    if not (mentions_currency or mentions_test_scores):
        return out

    # Collect citation texts
    evidence_texts = []
    for c in citations:
        t = (c.get("quote") or "").strip()
        if t:
            evidence_texts.append(t.lower())

    if not evidence_texts:
        out.append(
            Issue(
                code="NUMERIC_CLAIM_NO_CITATIONS",
                severity="ERROR",
                message="Answer includes numeric claims, but citations are missing.",
                details={
                    "mentions_currency": mentions_currency,
                    "mentions_test_scores": mentions_test_scores,
                },
            )
        )
        return out

    # Extract the actual numbers from answer 
    nums_in_answer = set(NUMBER_RE.findall(answer))
    # just check for any currency/test-score evidence
    if len(nums_in_answer) > 30:
        nums_in_answer = set(list(nums_in_answer)[:30])

    # Determine if at least ONE relevant numeric token appears in evidence
    evidence_blob = " ".join(evidence_texts)

    matched = 0
    for n in nums_in_answer:
        n_norm = n.replace(" ", "")
        if not n_norm:
            continue
        if n_norm in evidence_blob.replace(" ", ""):
            matched += 1

    # If currency/test scores are present but zero numeric overlap, flag
    if matched == 0:
        out.append(
            Issue(
                code="NUMERIC_CLAIM_UNSUPPORTED",
                severity="WARN",
                message="Answer includes numeric claims, but citations do not appear to contain the same numbers.",
                details={
                    "numbers_in_answer_sample": sorted(list(nums_in_answer))[:10],
                    "citations_count": len(citations),
                },
            )
        )

    # phone numbers should be directly cited if shown
    if mentions_phone:
        phone_ok = any(PHONE_RE.search(c.get("quote", "") or "") for c in citations)
        if not phone_ok:
            out.append(
                Issue(
                    code="PHONE_UNSUPPORTED",
                    severity="WARN",
                    message="Answer includes a phone number but citations do not show it.",
                )
            )

    return out


def _rule_idk_should_not_have_strong_evidence(answer: str, citations: Sequence[Dict[str, Any]]) -> List[Issue]:
    """
    If the answer says IDK but citations look strong/relevant, flag it:
    the system might be refusing incorrectly.
    """
    if IDK_PHRASE.lower() not in answer.lower():
        return []

    
    strong = []
    for c in citations:
        try:
            sc = float(c.get("score", 0.0))
        except Exception:
            sc = 0.0
        if sc >= 0.40:
            strong.append(sc)

    if strong:
        return [
            Issue(
                code="IDK_BUT_STRONG_CITATIONS",
                severity="WARN",
                message="Answer says 'I don't know' but there are strong citations; may be an overly conservative refusal.",
                details={"strong_scores": sorted(strong, reverse=True)[:5], "citations": len(citations)},
            )
        ]
    return []


def _rule_mode_sanity(query: str, mode: str) -> List[Issue]:
    """
    Prevent common UX failure: broad Q like "what is the tuition fee" being routed to list-mode.
    This doesn't block, but warns. You can later refine the router.
    """
    q = (query or "").strip().lower()
    if mode == "list":
        if "tuition" in q or "fee" in q or "fees" in q or "english" in q or "requirements" in q:
            return [
                Issue(
                    code="ROUTER_LIST_MAY_BE_WRONG",
                    severity="WARN",
                    message="Query looks like a Q&A (tuition/requirements), but router selected list-mode.",
                    details={"query": query},
                )
            ]
    return []



# Confidence heuristic

def _confidence_from_issues(issues: Sequence[Issue], citations: Sequence[Dict[str, Any]]) -> str:
    if any(i.severity == "ERROR" for i in issues):
        return "low"

    warn = sum(1 for i in issues if i.severity == "WARN")

    # If no citations, confidence low unless 
    if len(citations) == 0:
        return "low"

    if warn == 0:
        return "high"
    if warn <= 2:
        return "medium"
    return "low"
