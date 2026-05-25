from __future__ import annotations

from .schema import CONDITIONS, PilotItem


SYSTEM_PROMPT = (
    "You are a careful scientific reasoning assistant. Use only the provided passage as "
    "evidence. Preserve uncertainty. Return valid JSON only; no markdown fences."
)


def _label_instruction(dataset: str) -> str:
    if dataset == "scifact":
        return "The final_answer must be one of: SUPPORT, CONTRADICT, NOINFO."
    if dataset == "pubmedqa":
        return "The final_answer must be one of: yes, no, maybe."
    raise ValueError(f"Unknown dataset: {dataset}")


def _task_text(item: PilotItem) -> str:
    if item.dataset == "scifact":
        return (
            "Task: Decide whether the abstract SUPPORTS, CONTRADICTS, or gives NOINFO "
            "for the scientific claim.\n\n"
            f"Scientific claim:\n{item.claim_or_question}\n\n"
            f"Abstract/evidence text:\n{item.context}"
        )
    if item.dataset == "pubmedqa":
        return (
            "Task: Answer the biomedical question yes/no/maybe using only the provided "
            "context. Do not use outside knowledge.\n\n"
            f"Question:\n{item.claim_or_question}\n\n"
            f"Context:\n{item.context}"
        )
    raise ValueError(f"Unknown dataset: {item.dataset}")


def build_prompt(item: PilotItem, condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")

    label_instruction = _label_instruction(item.dataset)
    base = _task_text(item)

    if condition == "direct":
        response_contract = (
            'Return JSON with exactly these keys: "final_answer", "confidence", '
            '"brief_rationale". Keep brief_rationale to one sentence and do not include a '
            "reasoning trace."
        )
    elif condition == "cot":
        response_contract = (
            'Return JSON with exactly these keys: "reasoning_steps", "final_answer", '
            '"confidence". reasoning_steps must be an array of short numbered reasoning '
            "steps that cite what in the passage supports each step."
        )
    elif condition == "reflection":
        response_contract = (
            'Return JSON with exactly these keys: "initial_answer", "initial_reasoning_steps", '
            '"self_critique", "revised_reasoning_steps", "final_answer", "confidence". '
            "The critique should check whether each important claim is actually supported by "
            "the passage."
        )
    elif condition == "provenance":
        response_contract = (
            'Return JSON with exactly these keys: "tagged_claims", "reasoning_steps", '
            '"final_answer", "confidence". tagged_claims must be an array of objects with '
            '"claim", "origin_tag", and "used_for_final_answer". origin_tag must be one of '
            "[EVIDENCE], [INFERENCE], [ASSUMPTION], [WORLD], [CALCULATION]. Tag every "
            "intermediate claim."
        )
    else:
        response_contract = (
            'Return JSON with exactly these keys: "tagged_claims", "reasoning_steps", '
            '"quarantine_check", "final_answer", "confidence". tagged_claims must be an '
            'array of objects with "claim", "origin_tag", "verified_from_passage", and '
            '"used_for_final_answer". origin_tag must be one of [EVIDENCE], [INFERENCE], '
            "[ASSUMPTION], [WORLD], [CALCULATION]. Claims tagged [INFERENCE], [ASSUMPTION], "
            "or [WORLD] cannot be used as evidence for the final answer unless verified from "
            "the provided passage. quarantine_check should identify any quarantined claims."
        )

    return "\n\n".join([SYSTEM_PROMPT, base, label_instruction, response_contract])


def all_prompts_for_item(item: PilotItem) -> dict[str, str]:
    return {condition: build_prompt(item, condition) for condition in CONDITIONS}
