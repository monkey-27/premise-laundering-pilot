from __future__ import annotations

import json
import re
from typing import Any

from .schema import normalize_label

ORIGIN_RE = re.compile(r"\[(EVIDENCE|INFERENCE|ASSUMPTION|WORLD|CALCULATION)\]")
STEP_RE = re.compile(r"(?m)^\s*(?:\d+[\).:-]\s+|[-*]\s+)(.+?)\s*$")


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_output(raw_output: str) -> dict[str, Any]:
    data = extract_json_object(raw_output) or {}
    final = _find_final_answer(data, raw_output)
    steps = _find_steps(data, raw_output)
    claims = _find_claims(data, raw_output, steps)
    tags = sorted({tag for claim in claims for tag in [claim.get("declared_origin")] if tag})
    return {
        "json": data,
        "parsed_final_answer": final,
        "parsed_reasoning_steps": steps,
        "parsed_intermediate_claims": claims,
        "declared_origin_tags": tags,
    }


def _find_final_answer(data: dict[str, Any], raw_output: str) -> str | None:
    keys = ("final_answer", "answer", "final", "decision", "label")
    for key in keys:
        if key in data:
            normalized = normalize_label(str(data[key]))
            if normalized:
                return normalized

    match = re.search(
        r"(?:final[_\s-]?answer|answer|decision)\s*[:=-]\s*([A-Za-z_ ]+)",
        raw_output,
        flags=re.IGNORECASE,
    )
    if match:
        normalized = normalize_label(match.group(1))
        if normalized:
            return normalized

    for token in ("SUPPORT", "CONTRADICT", "NOINFO", "yes", "no", "maybe"):
        if re.search(rf"\b{re.escape(token)}\b", raw_output, flags=re.IGNORECASE):
            normalized = normalize_label(token)
            if normalized:
                return normalized
    return None


def _find_steps(data: dict[str, Any], raw_output: str) -> list[str]:
    for key in ("reasoning_steps", "revised_reasoning_steps", "initial_reasoning_steps", "steps"):
        value = data.get(key)
        if isinstance(value, list):
            return [stringify_step(step) for step in value if stringify_step(step)]
    matches = STEP_RE.findall(raw_output)
    return [match.strip() for match in matches[:20]]


def stringify_step(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("step", "claim", "text", "reasoning"):
            if key in value:
                return str(value[key]).strip()
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _find_claims(data: dict[str, Any], raw_output: str, steps: list[str]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    tagged = data.get("tagged_claims")
    if isinstance(tagged, list):
        for idx, item in enumerate(tagged, start=1):
            if isinstance(item, dict):
                text = str(item.get("claim") or item.get("text") or "").strip()
                tag = _normalize_origin(item.get("origin_tag") or item.get("declared_origin"))
                used = item.get("used_for_final_answer")
                verified = item.get("verified_from_passage")
            else:
                text = str(item).strip()
                tag = _first_origin(text)
                used = None
                verified = None
            if text:
                claims.append(
                    {
                        "claim_id": f"claim-{idx}",
                        "claim_text": text,
                        "declared_origin": tag,
                        "reasoning_step": None,
                        "used_for_final_answer": used,
                        "verified_from_passage": verified,
                    }
                )

    if not claims:
        claim_sources = steps or STEP_RE.findall(raw_output)
        for idx, step in enumerate(claim_sources, start=1):
            text = step.strip()
            if text:
                claims.append(
                    {
                        "claim_id": f"claim-{idx}",
                        "claim_text": text,
                        "declared_origin": _first_origin(text),
                        "reasoning_step": text,
                        "used_for_final_answer": None,
                        "verified_from_passage": None,
                    }
                )

    return claims


def _normalize_origin(value: Any) -> str | None:
    if value is None:
        return None
    match = ORIGIN_RE.search(str(value).upper())
    if match:
        return f"[{match.group(1)}]"
    return None


def _first_origin(text: str) -> str | None:
    match = ORIGIN_RE.search(text.upper())
    if match:
        return f"[{match.group(1)}]"
    return None
