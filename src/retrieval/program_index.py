from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.config.settings import Settings


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _pick(df: pd.DataFrame, *names: str) -> str:
    cols = set(df.columns)
    for n in names:
        if n in cols:
            return n
    return ""


def load_program_index(cfg: Settings) -> List[Dict]:
    """
    Returns a list of dicts (program index rows) used for name→program_id resolution.
    Works for both bachelor + masters excels.
    """
    rows: List[Dict] = []

    for scope, path in [("bachelor", cfg.bachelors_excel), ("master", cfg.masters_excel)]:
        if not path.exists():
            continue

        df = pd.read_excel(path)
        df = _norm_cols(df)

        c_pid = _pick(df, "program_id", "programid")
        c_name = _pick(df, "program_name", "programname")
        c_abbr = _pick(df, "program_abbreviation", "program_abbreviation_", "programabbr", "abbreviation")
        c_degree_level = _pick(df, "degree", "degree_level")
        c_degree_type = _pick(df, "degree_type")
        c_location = _pick(df, "location")

        for _, r in df.iterrows():
            pid = str(r.get(c_pid, "")).strip() if c_pid else ""
            name = str(r.get(c_name, "")).strip() if c_name else ""
            abbr = str(r.get(c_abbr, "")).strip() if c_abbr else ""
            degree_level = str(r.get(c_degree_level, "")).strip() if c_degree_level else scope
            degree_type = str(r.get(c_degree_type, "")).strip() if c_degree_type else ""
            location = str(r.get(c_location, "")).strip() if c_location else ""

            if not pid or not name:
                continue

            rows.append(
                {
                    "program_id": pid,
                    "program_name": name,
                    "program_abbreviation": abbr,
                    "degree_level": degree_level,
                    "degree_type": degree_type,
                    "location": location,
                    "scope": scope,
                }
            )

    return rows
