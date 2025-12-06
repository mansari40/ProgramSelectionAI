from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ProgramCandidate:
    program_id: str
    title: str
    score: float
    meta: Dict[str, str]


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = re.sub(r"[^\w\s&/]+", "", s)
    return s


def _contains_word(haystack: str, needle: str) -> bool:
    # word-boundary-ish containment for short tokens like "cs"
    if len(needle) <= 3:
        return re.search(rf"(^|\s){re.escape(needle)}(\s|$)", haystack) is not None
    return needle in haystack


def resolve_program_ids(
    query: str,
    program_index: List[Dict],
    max_results: int = 5,
    min_score: float = 0.35,
) -> List[ProgramCandidate]:
    """
    program_index: list of dicts. Each dict should contain at least:
      - program_id
      - program_name
      - program_abbreviation (optional)
      - degree_level (optional) e.g. bachelor/master
      - degree_type (optional) e.g. BSc/MSc/BA/MBA
      - location (optional) e.g. on_campus/online
    Returns ranked ProgramCandidate list.
    """
    qn = _norm(query)

    # Light parsing hints from the query
    wants_online = any(k in qn for k in ["online", "distance", "remote"])
    wants_oncampus = any(k in qn for k in ["on campus", "oncampus", "campus"])
    wants_bachelor = any(k in qn for k in ["bachelor", "undergrad", "undergraduate"])
    wants_master = any(k in qn for k in ["master", "graduate", "msc", "mba"])

    candidates: List[ProgramCandidate] = []

    for row in program_index:
        pid = str(row.get("program_id") or "").strip()
        name = str(row.get("program_name") or "").strip()
        abbr = str(row.get("program_abbreviation") or "").strip()
        degree_level = str(row.get("degree_level") or row.get("degree") or "").strip()
        degree_type = str(row.get("degree_type") or "").strip()
        location = str(row.get("location") or "").strip()

        if not pid or not name:
            continue

        name_n = _norm(name)
        abbr_n = _norm(abbr)
        degree_level_n = _norm(degree_level)
        degree_type_n = _norm(degree_type)
        location_n = _norm(location)

        score = 0.0

        # Strong matches
        if name_n and name_n in qn:
            score += 1.00
        else:
            # partial token overlap for program name
            tokens = [t for t in name_n.split(" ") if len(t) >= 4]
            hits = sum(1 for t in tokens if _contains_word(qn, t))
            if tokens:
                score += 0.65 * (hits / len(tokens))

        # Abbreviation helps a lot
        if abbr_n and _contains_word(qn, abbr_n):
            score += 0.75

        # Degree hints
        if wants_bachelor and ("bachelor" in degree_level_n or _contains_word(qn, "bsc") or _contains_word(qn, "ba")):
            score += 0.20
        if wants_master and ("master" in degree_level_n or _contains_word(qn, "msc") or _contains_word(qn, "mba")):
            score += 0.20

        # Location hints
        if wants_online and "online" in location_n:
            score += 0.20
        if wants_oncampus and ("on" in location_n and "campus" in location_n):
            score += 0.20

        meta = {
            "program_name": name,
            "program_abbreviation": abbr,
            "degree_level": degree_level,
            "degree_type": degree_type,
            "location": location,
        }

        candidates.append(ProgramCandidate(program_id=pid, title=name, score=score, meta=meta))

    # Rank + threshold
    candidates.sort(key=lambda c: c.score, reverse=True)
    filtered = [c for c in candidates if c.score >= min_score]

    return filtered[:max_results]
