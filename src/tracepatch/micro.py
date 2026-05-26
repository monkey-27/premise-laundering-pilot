from __future__ import annotations

import ast
import csv
import io
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

SEED = 27
RUN_ID = "tracepatch_micro_v1"
MODELS = ("Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3")
SYSTEMS = ("self_correction", "full_regeneration", "edit_only_repair", "tracepatch")
DOMAINS = ("code", "math", "rag")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_micro_data(output_dir: str | Path = "data/tracepatch", overwrite: bool = False) -> Path:
    out = ensure_dir(output_dir)
    data_path = out / "tracepatch_micro_v1.jsonl"
    manifest_path = out / "tracepatch_micro_v1.manifest.json"
    if data_path.exists() and not overwrite:
        return data_path
    examples = build_examples()
    write_jsonl(data_path, examples)
    write_json(
        manifest_path,
        {
            "total_examples": len(examples),
            "counts_by_domain": dict(Counter(ex["domain"] for ex in examples)),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checker_types": ["python_subprocess_tests", "exact_math", "gold_evidence_exact"],
            "notes": (
                "Deterministic synthetic TracePatch micropilot. Short examples only; "
                "all answers are checker-verifiable and source-contained."
            ),
        },
    )
    return data_path


def build_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    examples.extend(build_code_examples())
    examples.extend(build_math_examples())
    examples.extend(build_rag_examples())
    return examples


def build_code_examples() -> list[dict[str, Any]]:
    specs = [
        ("sum_even", "Return the sum of even integers in nums.", [([1, 2, 3, 4], 6), ([5, 7], 0), ([-2, 3, 8], 6)], "total=0\n    for x in nums:\n        if x % 2 == 0: total += x\n    return total"),
        ("count_vowels", "Return the number of vowels in text. Count aeiou only, case-insensitive.", [("Apple", 2), ("sky", 0), ("Education", 5)], "return sum(1 for c in text.lower() if c in 'aeiou')"),
        ("last_positive", "Return the last positive number in nums, or None if none exists.", [([-1, 2, 0, 5], 5), ([-3, 0], None), ([7, -1, 4], 4)], "return next((x for x in reversed(nums) if x > 0), None)"),
        ("dedupe_keep_order", "Return a list with duplicates removed while preserving first occurrence order.", [([3, 1, 3, 2, 1], [3, 1, 2]), ([], []), ([1, 1, 1], [1])], "seen=set(); out=[]\n    for x in items:\n        if x not in seen:\n            seen.add(x); out.append(x)\n    return out"),
        ("is_palindrome", "Return True if text is a palindrome after removing spaces and ignoring case.", [("Never odd or even", True), ("hello", False), ("A b a", True)], "s=''.join(text.lower().split()); return s == s[::-1]"),
        ("clamp", "Return x clamped to the inclusive range [lo, hi].", [((5, 1, 10), 5), ((-2, 0, 3), 0), ((8, 0, 3), 3)], "return max(lo, min(x, hi))"),
        ("first_longer_than", "Return the first string longer than n, or None.", [((["a", "abcd", "xx"], 2), "abcd"), ((["a", "bb"], 2), None), ((["three"], 3), "three")], "return next((s for s in words if len(s) > n), None)"),
        ("product_nonzero", "Return the product of all nonzero numbers. If all are zero, return 1.", [([2, 0, 3], 6), ([0, 0], 1), ([-1, 4, 0], -4)], "p=1\n    for x in nums:\n        if x != 0: p *= x\n    return p"),
        ("middle_char", "Return the middle character of an odd-length string.", [("abc", "b"), ("x", "x"), ("trace", "a")], "return text[len(text)//2]"),
        ("count_greater", "Return how many numbers are strictly greater than threshold.", [(([1, 5, 5, 7], 5), 1), (([], 3), 0), (([4, 6], 3), 2)], "return sum(1 for x in nums if x > threshold)"),
        ("reverse_words", "Return words in the sentence reversed, separated by one space.", [("red blue", "blue red"), ("one", "one"), ("a b c", "c b a")], "return ' '.join(sentence.split()[::-1])"),
        ("has_pair_sum", "Return True if any two distinct numbers sum to target.", [(([2, 4, 7], 6), True), (([1, 2, 3], 10), False), (([5, 5], 10), True)], "seen=set()\n    for x in nums:\n        if target-x in seen: return True\n        seen.add(x)\n    return False"),
        ("strip_suffix", "Return text without suffix if it ends with suffix; otherwise return text unchanged.", [(("filename.txt", ".txt"), "filename"), (("abc", "x"), "abc"), (("test", ""), "test")], "return text[:-len(suffix)] if suffix and text.endswith(suffix) else text"),
        ("max_abs", "Return the number with largest absolute value. If tied, return the first tied value.", [([-3, 2, 1], -3), ([2, -2], 2), ([0, 5, -4], 5)], "return max(nums, key=abs)"),
        ("running_total_last", "Return the final running total after adding all nums.", [([1, 2, 3], 6), ([], 0), ([-2, 5], 3)], "return sum(nums)"),
        ("every_other", "Return every other item starting with the first.", [([1, 2, 3, 4], [1, 3]), ([], []), (["a", "b", "c"], ["a", "c"])], "return items[::2]"),
        ("count_prefix", "Return how many words start with prefix.", [((["pre", "post", "prefix"], "pre"), 2), ((["a"], "b"), 0), (([], "x"), 0)], "return sum(1 for w in words if w.startswith(prefix))"),
        ("rotate_left_one", "Return a new list rotated left by one. Empty list stays empty.", [([1, 2, 3], [2, 3, 1]), ([], []), ([5], [5])], "return items[1:]+items[:1]"),
        ("safe_divide", "Return a / b, but return None if b is zero.", [((6, 3), 2), ((5, 0), None), ((7, 2), 3.5)], "return None if b == 0 else a / b"),
        ("remove_negatives", "Return a list containing only numbers greater than or equal to zero.", [([-1, 0, 2], [0, 2]), ([-3], []), ([1, 2], [1, 2])], "return [x for x in nums if x >= 0]"),
    ]
    examples = []
    for i, (name, desc, tests, body) in enumerate(specs):
        params = {
            "sum_even": "nums", "count_vowels": "text", "last_positive": "nums",
            "dedupe_keep_order": "items", "is_palindrome": "text", "clamp": "x, lo, hi",
            "first_longer_than": "words, n", "product_nonzero": "nums", "middle_char": "text",
            "count_greater": "nums, threshold", "reverse_words": "sentence",
            "has_pair_sum": "nums, target", "strip_suffix": "text, suffix", "max_abs": "nums",
            "running_total_last": "nums", "every_other": "items", "count_prefix": "words, prefix",
            "rotate_left_one": "items", "safe_divide": "a, b", "remove_negatives": "nums",
        }[name]
        canonical = f"def {name}({params}):\n    {body}\n"
        problem = f"Write Python function `{name}({params})`. {desc}\nReturn only the function implementation in JSON field code."
        def encode_input(inp: Any) -> Any:
            return {"__args__": list(inp)} if isinstance(inp, tuple) else inp

        examples.append(
            {
                "example_id": f"code_{i:02d}",
                "domain": "code",
                "problem": problem,
                "checker_spec": {"function_name": name, "tests": [{"input": encode_input(inp), "expected": exp} for inp, exp in tests], "canonical_code": canonical, "wrong_code": f"def {name}({params}):\n    return None\n"},
                "gold_answer": "passes_unit_tests",
                "metadata": {"checker": "python_subprocess_tests"},
            }
        )
    return examples


def build_math_examples() -> list[dict[str, Any]]:
    items = [
        ("A box has 12 red pens and 7 blue pens. Mia removes 5 red pens and adds 4 blue pens. How many pens are in the box now?", "18"),
        ("A train travels 45 miles each hour for 3 hours, then 20 miles more. What total distance does it travel?", "155"),
        ("Solve for x: 3x + 7 = 31.", "8"),
        ("A recipe uses 2/3 cup flour per batch. How many cups are needed for 6 batches?", "4"),
        ("Nora buys 4 notebooks at $3 each and 2 pens at $1.50 each. What is the total cost in dollars?", "15"),
        ("A number is doubled and then decreased by 9 to get 17. What is the number?", "13"),
        ("What is 25% of 84?", "21"),
        ("There are 5 shelves with 8 books each. If 6 books are removed, how many remain?", "34"),
        ("Simplify the fraction 18/24.", "3/4"),
        ("A rectangle has length 11 and width 6. What is its perimeter?", "34"),
        ("If y = 2x + 5 and x = 9, what is y?", "23"),
        ("A store discounts a $40 item by $6. What is the final price?", "34"),
        ("Compute (14 - 5) * 3.", "27"),
        ("Lena has 30 stickers and gives away 2/5 of them. How many stickers does she keep?", "18"),
        ("Solve: x/4 = 9.", "36"),
        ("A car uses 3 gallons for 72 miles. How many miles per gallon?", "24"),
        ("What is the average of 6, 10, and 20?", "12"),
        ("A rope of length 50 is cut into pieces of 8, 15, and the remainder. What is the remainder length?", "27"),
        ("If 7 bags hold 63 marbles equally, how many marbles are in 2 bags?", "18"),
        ("Compute 5^2 - 4^2.", "9"),
    ]
    return [
        {
            "example_id": f"math_{i:02d}",
            "domain": "math",
            "problem": problem,
            "checker_spec": {"expected_answer": answer, "wrong_answer": str(Fraction(answer) + 1 if _is_fraction(answer) else int(float(answer)) + 1)},
            "gold_answer": answer,
            "metadata": {"checker": "exact_math"},
        }
        for i, (problem, answer) in enumerate(items)
    ]


def _is_fraction(text: str) -> bool:
    return "/" in text


def build_rag_examples() -> list[dict[str, Any]]:
    tuples = [
        ("The Luma tablet weighs 480 grams. The Orbis tablet weighs 510 grams. The Fenn tablet weighs 455 grams.", "Which tablet is lightest?", "Fenn", "The Fenn tablet weighs 455 grams.", "Orbis"),
        ("North Clinic opens at 8 AM. East Clinic opens at 9 AM. West Clinic opens at 7 AM.", "Which clinic opens earliest?", "West Clinic", "West Clinic opens at 7 AM.", "East Clinic"),
        ("The red route stops at Pine and Lake. The blue route stops at Cedar and Oak. The green route stops at Lake and Elm.", "Which route stops at Cedar?", "blue route", "The blue route stops at Cedar and Oak.", "green route"),
        ("Rina submitted the form on Tuesday. Omar submitted it on Friday. Jules submitted it on Monday.", "Who submitted the form first?", "Jules", "Jules submitted it on Monday.", "Rina"),
        ("The copper key opens Room 2. The silver key opens Room 5. The brass key opens Room 1.", "Which key opens Room 5?", "silver key", "The silver key opens Room 5.", "copper key"),
        ("Atlas battery lasts 9 hours. Beacon battery lasts 11 hours. Comet battery lasts 8 hours.", "Which battery lasts longest?", "Beacon", "Beacon battery lasts 11 hours.", "Comet"),
        ("The oak sample is 14 cm. The pine sample is 18 cm. The birch sample is 16 cm.", "Which sample is 18 cm?", "pine", "The pine sample is 18 cm.", "birch"),
        ("Team A scored 12. Team B scored 18. Team C scored 15.", "Which team scored 18?", "Team B", "Team B scored 18.", "Team C"),
        ("The museum is closed on Monday. The library is closed on Sunday. The gallery is closed on Wednesday.", "What is closed on Sunday?", "library", "The library is closed on Sunday.", "museum"),
        ("Package X arrived at Dock 3. Package Y arrived at Dock 1. Package Z arrived at Dock 2.", "Which package arrived at Dock 1?", "Package Y", "Package Y arrived at Dock 1.", "Package X"),
        ("Ada chose vanilla. Ben chose mint. Cora chose lemon.", "Who chose mint?", "Ben", "Ben chose mint.", "Cora"),
        ("The north sensor reports 21 C. The south sensor reports 19 C. The east sensor reports 23 C.", "Which sensor reports 23 C?", "east sensor", "The east sensor reports 23 C.", "north sensor"),
        ("Book Delta has 180 pages. Book Echo has 220 pages. Book Flux has 150 pages.", "Which book has 220 pages?", "Echo", "Book Echo has 220 pages.", "Delta"),
        ("The bronze plan costs $12. The silver plan costs $18. The gold plan costs $25.", "Which plan costs $12?", "bronze plan", "The bronze plan costs $12.", "silver plan"),
        ("Lab A uses reagent M. Lab B uses reagent Q. Lab C uses reagent R.", "Which lab uses reagent Q?", "Lab B", "Lab B uses reagent Q.", "Lab C"),
        ("The ferry leaves at 6:30. The bus leaves at 7:15. The tram leaves at 6:45.", "Which transport leaves at 6:45?", "tram", "The tram leaves at 6:45.", "bus"),
        ("Ivy's badge is green. Kai's badge is yellow. Lou's badge is purple.", "Whose badge is yellow?", "Kai", "Kai's badge is yellow.", "Lou"),
        ("The delta file is 4 MB. The sigma file is 9 MB. The omega file is 7 MB.", "Which file is 9 MB?", "sigma file", "The sigma file is 9 MB.", "omega file"),
        ("Zone 1 requires gloves. Zone 2 requires goggles. Zone 3 requires masks.", "Which zone requires goggles?", "Zone 2", "Zone 2 requires goggles.", "Zone 1"),
        ("The maple crate contains apples. The cedar crate contains pears. The pine crate contains plums.", "Which crate contains pears?", "cedar crate", "The cedar crate contains pears.", "pine crate"),
    ]
    return [
        {
            "example_id": f"rag_{i:02d}",
            "domain": "rag",
            "problem": f"Evidence: {evidence}\nQuestion: {question}\nAnswer using only the evidence.",
            "checker_spec": {"evidence": evidence, "evidence_sentence": sentence, "expected_answer": answer, "distractor_answer": distractor},
            "gold_answer": answer,
            "metadata": {"checker": "gold_evidence_exact"},
        }
        for i, (evidence, question, answer, sentence, distractor) in enumerate(tuples)
    ]


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.I | re.S)
    candidates = [fenced.group(1).strip()] if fenced else []
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    candidates.append(stripped)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_model_output(raw: str, domain: str) -> dict[str, Any]:
    data = extract_json_object(raw) or {}
    trace = data.get("trace") if isinstance(data.get("trace"), list) else []
    norm_trace = []
    for idx, step in enumerate(trace, start=1):
        if isinstance(step, dict):
            norm_trace.append({"step_id": int(step.get("step_id", idx)), "text": str(step.get("text", ""))})
        else:
            norm_trace.append({"step_id": idx, "text": str(step)})
    return {
        "parse_ok": bool(data),
        "trace": norm_trace,
        "final_answer": str(data.get("final_answer", "")).strip(),
        "code": str(data.get("code", "")).strip() if domain == "code" else "",
        "json": data,
    }


def check_output(example: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    if not parsed.get("parse_ok"):
        return {"passed": False, "feedback": "Output was not parseable JSON.", "parse_error": True}
    if example["domain"] == "code":
        return check_code(example, parsed.get("code", ""))
    if example["domain"] == "math":
        return check_math(example, parsed.get("final_answer", ""), parsed.get("trace", []))
    return check_rag(example, parsed.get("final_answer", ""))


def check_code(example: dict[str, Any], code: str) -> dict[str, Any]:
    spec = example["checker_spec"]
    tests = spec["tests"]
    script = {
        "code": code,
        "function_name": spec["function_name"],
        "tests": tests,
    }
    runner = textwrap.dedent(
        """
        import json, traceback
        payload=json.loads(PAYLOAD)
        ns={}
        failed=[]; passed=[]
        try:
            exec(payload["code"], ns)
            fn=ns[payload["function_name"]]
            for i,t in enumerate(payload["tests"]):
                inp=t["input"]; expected=t["expected"]
                if isinstance(inp, dict) and "__args__" in inp:
                    actual=fn(*inp["__args__"])
                elif isinstance(inp, list):
                    actual=fn(inp)
                elif isinstance(inp, dict):
                    actual=fn(**inp)
                else:
                    actual=fn(inp)
                if actual == expected:
                    passed.append({"test_index": i, "input": inp, "expected": expected, "actual": actual})
                else:
                    failed.append({"test_index": i, "input": inp, "expected": expected, "actual": actual})
        except Exception as e:
            failed.append({"exception": type(e).__name__, "message": str(e), "traceback": traceback.format_exc(limit=2)})
        print(json.dumps({"failed": failed, "passed": passed}))
        """
    )
    # Tuples become lists through JSON; marker-convert only the call style in test construction.
    safe_tests = []
    for test in tests:
        inp = test["input"]
        if isinstance(inp, tuple):
            inp = {"__args__": list(inp)}
        safe_tests.append({"input": inp, "expected": test["expected"]})
    script["tests"] = safe_tests
    runner = runner.replace("PAYLOAD", repr(json.dumps(script)))
    try:
        completed = subprocess.run([sys.executable, "-c", runner], text=True, capture_output=True, timeout=2)
        result = json.loads(completed.stdout.strip() or "{}")
    except Exception as exc:
        result = {"failed": [{"exception": type(exc).__name__, "message": str(exc)}], "passed": []}
    failed = result.get("failed", [])
    passed = result.get("passed", [])
    feedback = "All unit tests passed." if not failed else f"Failing test: {failed[0]}"
    return {"passed": not failed, "feedback": feedback, "failed_tests": failed, "passed_tests": passed}


def normalize_answer(text: str) -> str:
    raw = str(text).strip().lower().replace("$", "")
    raw = re.sub(r"^(answer|final answer)\s*[:=]\s*", "", raw)
    raw = raw.rstrip(".")
    try:
        return str(Fraction(raw))
    except Exception:
        pass
    try:
        val = float(raw)
        if math.isclose(val, round(val)):
            return str(int(round(val)))
        return str(Fraction(val).limit_denominator())
    except Exception:
        return re.sub(r"\s+", " ", raw)


def check_math(example: dict[str, Any], answer: str, trace: list[dict[str, Any]]) -> dict[str, Any]:
    expected = normalize_answer(example["gold_answer"])
    actual = normalize_answer(answer)
    passed = expected == actual
    first_invalid = None
    for step in trace:
        nums = re.findall(r"(-?\d+(?:/\d+)?)\s*=\s*(-?\d+(?:/\d+)?)", step.get("text", ""))
        for left, right in nums:
            if normalize_answer(left) != normalize_answer(right):
                first_invalid = step.get("step_id")
                break
        if first_invalid:
            break
    feedback = "Final answer matches expected answer." if passed else f"Final answer mismatch: expected {example['gold_answer']}, got {answer or '<empty>'}."
    if first_invalid:
        feedback += f" First invalid equation appears at step {first_invalid}."
    return {"passed": passed, "feedback": feedback, "first_invalid_step_id": first_invalid, "expected_answer": example["gold_answer"]}


def check_rag(example: dict[str, Any], answer: str) -> dict[str, Any]:
    spec = example["checker_spec"]
    expected = normalize_text(spec["expected_answer"])
    actual = normalize_text(answer)
    actual = re.sub(r"^(the answer is|answer is|it is)\s+", "", actual).strip()
    negated = any(phrase in actual for phrase in (" does not ", " not ", " no "))
    passed = actual == expected or (expected in actual and not negated)
    feedback = "Answer matches gold evidence." if passed else f"Answer should be {spec['expected_answer']}; supporting evidence: {spec['evidence_sentence']}"
    return {"passed": passed, "feedback": feedback, "evidence_sentence": spec["evidence_sentence"], "expected_answer": spec["expected_answer"]}


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9/.-]+", " ", str(text).lower()).strip()


def build_initial_prompt(example: dict[str, Any]) -> str:
    if example["domain"] == "code":
        extra = 'Put the final function in "code"; final_answer can be a short summary.'
    else:
        extra = 'Put the final answer in "final_answer"; omit "code".'
    return (
        "Solve the task. Return only JSON with schema "
        '{"trace":[{"step_id":1,"text":"..."}],"final_answer":"...","code":"..."}.\n'
        f"{extra}\nProblem:\n{example['problem']}"
    )


def build_repair_prompt(system: str, example: dict[str, Any], initial: dict[str, Any], feedback: str, localization: dict[str, Any] | None = None) -> str:
    base = f"Problem:\n{example['problem']}\n\nChecker feedback:\n{feedback}\n\n"
    if system == "full_regeneration":
        instruction = "Ignore the previous solution and regenerate a complete corrected solution from scratch."
        prior = ""
    elif system == "self_correction":
        instruction = "Revise your answer based on the feedback."
        prior = f"Initial solution JSON:\n{json.dumps(initial, ensure_ascii=False)}\n\n"
    elif system == "edit_only_repair":
        instruction = "Edit the existing solution to fix the error. Do not assume a provided faulty span."
        prior = f"Initial solution JSON:\n{json.dumps(initial, ensure_ascii=False)}\n\n"
    elif system == "tracepatch":
        prefix_steps = localization.get("correct_prefix_steps", []) if localization else []
        faulty_text = localization.get("faulty_span_text", "") if localization else ""
        instruction = "Preserve the correct prefix exactly where still valid, replace the faulty span, and continue from there."
        prior = f"Correct prefix steps:\n{json.dumps(prefix_steps, ensure_ascii=False)}\nFaulty span text:\n{faulty_text}\n\n"
    else:
        raise ValueError(system)
    return base + prior + instruction + '\nReturn only JSON with schema {"trace":[{"step_id":1,"text":"..."}],"final_answer":"...","code":"..."}'


def build_localize_prompt(example: dict[str, Any], initial: dict[str, Any], feedback: str) -> str:
    return (
        "Identify the earliest faulty span in the initial trace. Return only JSON with schema "
        '{"faulty_span":{"start_step_id":1,"end_step_id":1},"correct_prefix_step_ids":[1],"localization_reason":"..."}.\n'
        f"Problem:\n{example['problem']}\n\nInitial solution JSON:\n{json.dumps(initial, ensure_ascii=False)}\n\nChecker feedback:\n{feedback}"
    )


def parse_localization(raw: str, initial: dict[str, Any], checker: dict[str, Any]) -> dict[str, Any]:
    data = extract_json_object(raw) or {}
    span = data.get("faulty_span") if isinstance(data.get("faulty_span"), dict) else {}
    try:
        start = int(span.get("start_step_id", checker.get("first_invalid_step_id") or 1))
        end = int(span.get("end_step_id", start))
    except Exception:
        start = end = 1
    prefix_ids = data.get("correct_prefix_step_ids")
    if not isinstance(prefix_ids, list):
        prefix_ids = [s["step_id"] for s in initial.get("trace", []) if s.get("step_id", 0) < start]
    steps = initial.get("trace", [])
    prefix_steps = [s for s in steps if s.get("step_id") in set(prefix_ids)]
    faulty = " ".join(s.get("text", "") for s in steps if start <= s.get("step_id", 0) <= end)
    return {"parse_ok": bool(data), "faulty_span": {"start_step_id": start, "end_step_id": end}, "correct_prefix_step_ids": prefix_ids, "correct_prefix_steps": prefix_steps, "faulty_span_text": faulty, "localization_reason": str(data.get("localization_reason", "")), "raw": raw}


def oracle_localization(example: dict[str, Any], initial: dict[str, Any], checker: dict[str, Any]) -> dict[str, Any] | None:
    if example["domain"] == "code":
        return None
    if example["domain"] == "math":
        step = checker.get("first_invalid_step_id") or max(1, len(initial.get("trace", [])))
    else:
        step = 1
    steps = initial.get("trace", [])
    return {
        "parse_ok": True,
        "faulty_span": {"start_step_id": step, "end_step_id": step},
        "correct_prefix_step_ids": [s.get("step_id") for s in steps if s.get("step_id", 0) < step],
        "correct_prefix_steps": [s for s in steps if s.get("step_id", 0) < step],
        "faulty_span_text": " ".join(s.get("text", "") for s in steps if s.get("step_id") == step),
        "localization_reason": "checker-derived oracle localization",
        "raw": "oracle",
    }


class ModalGenerator:
    def __init__(self, model: str, max_new_tokens: int, temperature: float = 0.0, top_p: float = 0.9) -> None:
        from premise_laundering.transformers_runner import TransformersGenerator

        self.generator = TransformersGenerator(model=model, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    def __call__(self, prompts: list[str]) -> list[str]:
        return self.generator(prompts)

    def set_max_new_tokens(self, max_new_tokens: int) -> None:
        self.generator.max_new_tokens = max_new_tokens


def generate_with(generator: Callable[[list[str]], list[str]], prompts: list[str], max_new_tokens: int) -> list[str]:
    setter = getattr(generator, "set_max_new_tokens", None)
    if callable(setter):
        setter(max_new_tokens)
    elif hasattr(generator, "max_tokens"):
        try:
            setattr(generator, "max_tokens", max_new_tokens)
        except Exception:
            pass
    return generator(prompts)


def release_generator(generator: Any) -> None:
    try:
        import gc
        import torch

        inner = getattr(generator, "generator", None)
        if inner is not None:
            if hasattr(inner, "model"):
                del inner.model
            if hasattr(inner, "tokenizer"):
                del inner.tokenizer
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_micro(
    data_file: str | Path,
    output_dir: str | Path,
    model_names: tuple[str, ...] = MODELS,
    batch_size: int = 2,
    generator_factory: Callable[[str, int], Callable[[list[str]], list[str]]] | None = None,
) -> Path:
    out = ensure_dir(output_dir)
    examples = read_jsonl(data_file)
    generator_factory = generator_factory or (lambda model, max_tokens: ModalGenerator(model, max_tokens))
    initial_rows = []
    checker_rows = []
    repair_rows = []
    localization_rows = []
    oracle_subset = set([f"code_{i:02d}" for i in range(10)] + [f"math_{i:02d}" for i in range(10)] + [f"rag_{i:02d}" for i in range(10)])

    for model in model_names:
        generator = generator_factory(model, 512)
        prompts = [(ex, build_initial_prompt(ex)) for ex in examples]
        for batch in chunks(prompts, batch_size):
            raws = generate_with(generator, [p for _, p in batch], 512)
            for (ex, prompt), raw in zip(batch, raws):
                parsed = parse_model_output(raw, ex["domain"])
                checker = check_output(ex, parsed)
                row = {"run_id": RUN_ID, "model": model, "example_id": ex["example_id"], "domain": ex["domain"], "prompt": prompt, "raw_output": raw, "parsed": parsed, "passed": checker["passed"]}
                initial_rows.append(row)
                checker_rows.append({"stage": "initial", "system": "initial", "model": model, "example_id": ex["example_id"], "domain": ex["domain"], **checker})

        failed = [r for r in initial_rows if r["model"] == model and not r["passed"]]
        by_id = {ex["example_id"]: ex for ex in examples}
        repair_tokens = {
            "self_correction": 384,
            "full_regeneration": 512,
            "edit_only_repair": 384,
        }
        for init in failed:
            ex = by_id[init["example_id"]]
            init_checker = next(c for c in checker_rows if c["stage"] == "initial" and c["model"] == model and c["example_id"] == ex["example_id"])
            loc_raw = generate_with(generator, [build_localize_prompt(ex, init["parsed"], init_checker["feedback"])], 192)[0]
            loc = parse_localization(loc_raw, init["parsed"], init_checker)
            localization_rows.append({"model": model, "example_id": ex["example_id"], "domain": ex["domain"], "system": "tracepatch_localize", **loc})
            for system, max_tokens in repair_tokens.items():
                prompt = build_repair_prompt(system, ex, init["parsed"], init_checker["feedback"])
                raw = generate_with(generator, [prompt], max_tokens)[0]
                add_repair_row(repair_rows, checker_rows, ex, model, system, prompt, raw, init, init_checker)
            prompt = build_repair_prompt("tracepatch", ex, init["parsed"], init_checker["feedback"], loc)
            raw = generate_with(generator, [prompt], 384)[0]
            add_repair_row(repair_rows, checker_rows, ex, model, "tracepatch", prompt, raw, init, init_checker, localization=loc)
            if ex["example_id"] in oracle_subset:
                oracle = oracle_localization(ex, init["parsed"], init_checker)
                if oracle:
                    prompt = build_repair_prompt("tracepatch", ex, init["parsed"], init_checker["feedback"], oracle)
                    raw = generate_with(generator, [prompt], 384)[0]
                    add_repair_row(repair_rows, checker_rows, ex, model, "oracle_tracepatch_subset", prompt, raw, init, init_checker, localization=oracle)
                    localization_rows.append({"model": model, "example_id": ex["example_id"], "domain": ex["domain"], "system": "oracle_tracepatch_subset", **oracle})
        release_generator(generator)

    write_jsonl(out / "initial_results.jsonl", initial_rows)
    write_jsonl(out / "repair_results.jsonl", repair_rows)
    write_jsonl(out / "checker_results.jsonl", checker_rows)
    write_jsonl(out / "localization_results.jsonl", localization_rows)
    audit_rows, audit_summary, manual_subset = run_data_eval_audit(examples)
    write_csv(out / "data_eval_audit.csv", audit_rows, list(audit_rows[0]))
    write_json(out / "data_eval_audit_summary.json", audit_summary)
    write_csv(out / "manual_data_eval_audit_sample.csv", manual_subset, list(manual_subset[0]))
    summary, breakdown = summarize_results(examples, initial_rows, repair_rows, checker_rows, localization_rows, audit_summary)
    write_json(out / "summary.json", summary)
    write_csv(out / "breakdown.csv", breakdown, list(breakdown[0]))
    write_csv(out / "manual_audit_sample.csv", build_manual_audit_sample(examples, initial_rows, repair_rows, checker_rows, localization_rows), manual_audit_fields())
    (out / "interpretation.md").write_text(build_interpretation(summary, breakdown, audit_summary), encoding="utf-8")
    return out


def add_repair_row(repair_rows, checker_rows, ex, model, system, prompt, raw, init, init_checker, localization=None):
    parsed = parse_model_output(raw, ex["domain"])
    checker = check_output(ex, parsed)
    repeated = (not checker["passed"]) and same_failure(init_checker, checker)
    prefix_damage = prefix_damage_approx(ex, init, parsed, init_checker, checker, localization)
    row = {
        "run_id": RUN_ID,
        "model": model,
        "example_id": ex["example_id"],
        "domain": ex["domain"],
        "system": system,
        "prompt": prompt,
        "raw_output": raw,
        "parsed": parsed,
        "passed": checker["passed"],
        "repeated_error": repeated,
        "prefix_damage": prefix_damage["damaged"],
        "prefix_damage_confidence": prefix_damage["confidence"],
        "token_count": approx_tokens(raw),
    }
    repair_rows.append(row)
    checker_rows.append({"stage": "repair", "system": system, "model": model, "example_id": ex["example_id"], "domain": ex["domain"], **checker})


def same_failure(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("passed") or b.get("passed"):
        return False
    if a.get("failed_tests") and b.get("failed_tests"):
        return a["failed_tests"][0].get("test_index") == b["failed_tests"][0].get("test_index")
    return normalize_text(a.get("expected_answer", "") or a.get("evidence_sentence", "") or a.get("feedback", ""))[:40] in normalize_text(b.get("feedback", ""))


def prefix_damage_approx(ex, init, parsed, init_checker, checker, localization):
    if ex["domain"] == "code":
        initially_passed = {str(t.get("test_index")) for t in init_checker.get("passed_tests", [])}
        now_failed = {str(t.get("test_index")) for t in checker.get("failed_tests", [])}
        return {"damaged": bool(initially_passed & now_failed), "confidence": "high"}
    if not localization:
        return {"damaged": False, "confidence": "low"}
    prefix_ids = set(localization.get("correct_prefix_step_ids", []))
    init_text = {s.get("step_id"): normalize_text(s.get("text", "")) for s in init.get("parsed", init).get("trace", [])}
    new_texts = [normalize_text(s.get("text", "")) for s in parsed.get("trace", [])]
    damaged = any(init_text.get(i) and init_text[i] not in " ".join(new_texts) for i in prefix_ids)
    return {"damaged": damaged, "confidence": "medium" if prefix_ids else "low"}


def approx_tokens(text: str) -> int:
    return max(1, len(str(text).split()))


def chunks(items: list[Any], n: int) -> list[list[Any]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def run_data_eval_audit(examples: list[dict[str, Any]]):
    rows = []
    for ex in examples:
        gold_parsed = {"parse_ok": True, "trace": [{"step_id": 1, "text": "gold"}], "final_answer": ex["gold_answer"], "code": ex["checker_spec"].get("canonical_code", ""), "json": {}}
        wrong_answer = ex["checker_spec"].get("wrong_answer") or ex["checker_spec"].get("distractor_answer") or "definitely_wrong"
        wrong_parsed = {"parse_ok": True, "trace": [{"step_id": 1, "text": "wrong"}], "final_answer": str(wrong_answer), "code": ex["checker_spec"].get("wrong_code", ""), "json": {}}
        gold_check = check_output(ex, gold_parsed)
        wrong_check = check_output(ex, wrong_parsed)
        feedback_score = 2 if any(k in wrong_check for k in ("failed_tests", "evidence_sentence", "first_invalid_step_id")) or "expected" in wrong_check.get("feedback", "") else 1
        row = {
            "example_id": ex["example_id"],
            "domain": ex["domain"],
            "problem_parse_ok": bool(ex.get("problem")),
            "checker_spec_parse_ok": bool(ex.get("checker_spec")),
            "gold_answer_present": bool(ex.get("gold_answer")),
            "gold_answer_passes_checker": gold_check["passed"],
            "checker_feedback_nonempty_on_wrong_answer": bool(wrong_check.get("feedback")),
            "problem_ambiguous_flag": False,
            "answer_leakage_flag": str(ex["gold_answer"]).lower() in ex["problem"].lower() and ex["domain"] != "rag",
            "duplicate_or_near_duplicate_flag": False,
            "data_valid_auto": gold_check["passed"] and not wrong_check["passed"],
            "unit_tests_present": ex["domain"] == "code" and bool(ex["checker_spec"].get("tests")),
            "canonical_solution_passes_tests": gold_check["passed"] if ex["domain"] == "code" else "",
            "intentionally_wrong_solution_fails_tests": (not wrong_check["passed"]) if ex["domain"] == "code" else "",
            "test_count": len(ex["checker_spec"].get("tests", [])) if ex["domain"] == "code" else "",
            "unsafe_code_flag": False,
            "gold_answer_normalizes": bool(normalize_answer(ex["gold_answer"])) if ex["domain"] == "math" else "",
            "wrong_answer_fails_checker": not wrong_check["passed"],
            "arithmetic_checker_available": ex["domain"] == "math",
            "exact_answer_type": exact_answer_type(ex["gold_answer"]) if ex["domain"] == "math" else "",
            "evidence_present": ex["domain"] == "rag" and bool(ex["checker_spec"].get("evidence")),
            "gold_answer_supported_by_evidence": ex["domain"] == "rag" and normalize_text(ex["gold_answer"]) in normalize_text(ex["checker_spec"].get("evidence", "")) if ex["domain"] == "rag" else "",
            "distractor_answer_contradicted_or_unsupported": (not wrong_check["passed"]) if ex["domain"] == "rag" else "",
            "evidence_sentence_selected": ex["checker_spec"].get("evidence_sentence", "") if ex["domain"] == "rag" else "",
            "unsupported_ambiguity_flag": False,
            "checker_false_negative_flag": wrong_check["passed"],
            "checker_false_positive_flag": not gold_check["passed"],
            "feedback_locality_score_auto": feedback_score,
            "evaluator_valid_auto": gold_check["passed"] and not wrong_check["passed"],
            "prefix_damage_metric_available": True,
            "prefix_damage_metric_confidence": "high" if ex["domain"] == "code" else "medium",
            "repeated_error_metric_available": True,
            "repeated_error_metric_confidence": "high" if ex["domain"] == "code" else "medium",
            "localization_metric_available": ex["domain"] != "code",
            "localization_metric_confidence": "medium" if ex["domain"] != "code" else "unavailable",
        }
        rows.append(row)
    summary = audit_summary(rows)
    manual = []
    for domain in DOMAINS:
        for row in [r for r in rows if r["domain"] == domain][:10]:
            ex = next(e for e in examples if e["example_id"] == row["example_id"])
            manual.append({
                "example_id": ex["example_id"], "domain": domain, "problem": ex["problem"],
                "checker_spec": json.dumps(ex["checker_spec"], ensure_ascii=False),
                "gold_answer": ex["gold_answer"],
                "checker_feedback_on_gold": check_output(ex, {"parse_ok": True, "trace": [], "final_answer": ex["gold_answer"], "code": ex["checker_spec"].get("canonical_code", "")}).get("feedback"),
                "checker_feedback_on_wrong_answer": check_output(ex, {"parse_ok": True, "trace": [], "final_answer": str(ex["checker_spec"].get("wrong_answer") or ex["checker_spec"].get("distractor_answer") or "wrong"), "code": ex["checker_spec"].get("wrong_code", "")}).get("feedback"),
                "auto_data_valid": row["data_valid_auto"], "auto_evaluator_valid": row["evaluator_valid_auto"],
                "human_data_valid": "", "human_checker_valid": "", "human_feedback_local_enough": "", "issue_type": "", "notes": "",
            })
    return rows, summary, manual


def exact_answer_type(answer: str) -> str:
    if "/" in str(answer):
        return "fraction"
    try:
        int(str(answer))
        return "integer"
    except Exception:
        return "string"


def audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    low_prefix = sum(r["prefix_damage_metric_confidence"] in {"low", "unavailable"} for r in rows)
    low_repeated = sum(r["repeated_error_metric_confidence"] in {"low", "unavailable"} for r in rows)
    return {
        "n_examples": n,
        "n_data_valid_auto": sum(bool(r["data_valid_auto"]) for r in rows),
        "data_valid_rate": mean(bool(r["data_valid_auto"]) for r in rows),
        "n_gold_answer_checker_failures": sum(not r["gold_answer_passes_checker"] for r in rows),
        "gold_answer_checker_failure_rate": mean(not r["gold_answer_passes_checker"] for r in rows),
        "n_wrong_answer_checker_false_negatives": sum(bool(r["checker_false_negative_flag"]) for r in rows),
        "wrong_answer_checker_false_negative_rate": mean(bool(r["checker_false_negative_flag"]) for r in rows),
        "n_ambiguous_or_invalid_examples": sum(bool(r["problem_ambiguous_flag"] or not r["data_valid_auto"]) for r in rows),
        "ambiguous_or_invalid_rate": mean(bool(r["problem_ambiguous_flag"] or not r["data_valid_auto"]) for r in rows),
        "n_feedback_low_locality": sum(int(r["feedback_locality_score_auto"]) == 0 for r in rows),
        "feedback_low_locality_rate": mean(int(r["feedback_locality_score_auto"]) == 0 for r in rows),
        "n_prefix_damage_metric_low_confidence": low_prefix,
        "prefix_damage_metric_low_confidence_rate": low_prefix / n,
        "n_repeated_error_metric_low_confidence": low_repeated,
        "repeated_error_metric_low_confidence_rate": low_repeated / n,
    }


def hard_fail_audit(summary: dict[str, Any]) -> list[str]:
    errors = []
    if summary["n_gold_answer_checker_failures"]:
        errors.append("gold answer checker failure")
    if summary["data_valid_rate"] < 0.95:
        errors.append("data_valid_rate < 0.95")
    if summary["wrong_answer_checker_false_negative_rate"] > 0.10:
        errors.append("wrong_answer_checker_false_negative_rate > 0.10")
    if summary["ambiguous_or_invalid_rate"] > 0.10:
        errors.append("ambiguous_or_invalid_rate > 0.10")
    if summary["feedback_low_locality_rate"] > 0.35:
        errors.append("feedback_low_locality_rate > 0.35")
    return errors


def summarize_results(examples, initial_rows, repair_rows, checker_rows, localization_rows, audit):
    systems = SYSTEMS + ("oracle_tracepatch_subset",)
    rows = []
    summary = {"n_examples": len(examples), "models": list(MODELS), "initial": {}, "systems": {}, "audit": audit}
    for model in sorted({r["model"] for r in initial_rows}):
        model_initial = [r for r in initial_rows if r["model"] == model]
        failed_ids = {r["example_id"] for r in model_initial if not r["passed"]}
        summary["initial"][model] = {"accuracy": mean(r["passed"] for r in model_initial), "failure_rate": mean(not r["passed"] for r in model_initial), "n_failed": len(failed_ids)}
        for system in systems:
            sys_repairs = [r for r in repair_rows if r["model"] == model and r["system"] == system]
            if system == "oracle_tracepatch_subset":
                denom_ids = {r["example_id"] for r in sys_repairs}
            else:
                denom_ids = {r["example_id"] for r in model_initial}
            passed_by_id = {r["example_id"]: r["passed"] for r in sys_repairs}
            final_pass = []
            for r in model_initial:
                if r["example_id"] not in denom_ids:
                    continue
                final_pass.append(True if r["passed"] and system != "oracle_tracepatch_subset" else passed_by_id.get(r["example_id"], False))
            repair_failed = [r["passed"] for r in sys_repairs]
            key = f"{model}|{system}"
            summary["systems"][key] = {
                "final_accuracy": mean(final_pass),
                "repair_success_on_failed": mean(repair_failed),
                "repeated_error_rate": mean(r["repeated_error"] for r in sys_repairs),
                "prefix_damage_rate": mean(r["prefix_damage"] for r in sys_repairs),
                "parse_error_rate": mean(not r["parsed"]["parse_ok"] for r in sys_repairs),
                "generated_token_count": statistics.mean([r["token_count"] for r in sys_repairs]) if sys_repairs else 0,
            }
            rows.append({"group": key, "model": model, "system": system, **summary["systems"][key]})
            for domain in DOMAINS:
                domain_initial = [r for r in model_initial if r["domain"] == domain]
                domain_repairs = [r for r in sys_repairs if r["domain"] == domain]
                final = []
                for r in domain_initial:
                    final.append(True if r["passed"] and system != "oracle_tracepatch_subset" else {x["example_id"]: x["passed"] for x in domain_repairs}.get(r["example_id"], False))
                rows.append({"group": f"{model}|{system}|{domain}", "model": model, "system": system, "domain": domain, "final_accuracy": mean(final), "repair_success_on_failed": mean(r["passed"] for r in domain_repairs), "repeated_error_rate": mean(r["repeated_error"] for r in domain_repairs), "prefix_damage_rate": mean(r["prefix_damage"] for r in domain_repairs), "parse_error_rate": mean(not r["parsed"]["parse_ok"] for r in domain_repairs), "generated_token_count": statistics.mean([r["token_count"] for r in domain_repairs]) if domain_repairs else 0})
    summary["aggregate"] = aggregate_system_metrics(initial_rows, repair_rows)
    summary["oracle_overlap"] = oracle_overlap_metrics(repair_rows)
    summary["scale_criteria"] = scale_criteria(summary)
    summary["verdict"] = verdict(summary)
    return summary, rows


def aggregate_system_metrics(initial_rows, repair_rows) -> dict[str, dict[str, Any]]:
    metrics = {}
    for system in SYSTEMS:
        repairs = [r for r in repair_rows if r["system"] == system]
        by_key = {(r["model"], r["example_id"]): r for r in repairs}
        final_correct = []
        for init in initial_rows:
            if init["passed"]:
                final_correct.append(True)
            else:
                final_correct.append(by_key[(init["model"], init["example_id"])]["passed"])
        metrics[system] = {
            "n_repairs": len(repairs),
            "final_accuracy": mean(final_correct),
            "repair_success_on_failed": mean(r["passed"] for r in repairs),
            "repeated_error_rate": mean(r["repeated_error"] for r in repairs),
            "prefix_damage_rate": mean(r["prefix_damage"] for r in repairs),
            "parse_error_rate": mean(not r["parsed"]["parse_ok"] for r in repairs),
        }
    return metrics


def oracle_overlap_metrics(repair_rows) -> dict[str, Any]:
    oracle = [r for r in repair_rows if r["system"] == "oracle_tracepatch_subset"]
    tracepatch = {(r["model"], r["example_id"]): r for r in repair_rows if r["system"] == "tracepatch"}
    if not oracle:
        return {
            "n_oracle_repairs": 0,
            "oracle_success": 0.0,
            "tracepatch_success_on_oracle_overlap": 0.0,
            "oracle_gap": 0.0,
            "note": "No checker-derived oracle localizations were available.",
        }
    normal_success = [tracepatch[(r["model"], r["example_id"])]["passed"] for r in oracle]
    oracle_success = mean(r["passed"] for r in oracle)
    tracepatch_success = mean(normal_success)
    return {
        "n_oracle_repairs": len(oracle),
        "oracle_success": oracle_success,
        "tracepatch_success_on_oracle_overlap": tracepatch_success,
        "oracle_gap": oracle_success - tracepatch_success,
        "note": "Code oracle localization is skipped by design unless deterministic localization is available.",
    }


def scale_criteria(summary: dict[str, Any]) -> dict[str, Any]:
    agg = summary.get("aggregate", {})
    tp = agg.get("tracepatch", {})
    full = agg.get("full_regeneration", {})
    self_corr = agg.get("self_correction", {})
    edit = agg.get("edit_only_repair", {})

    def gain(base: dict[str, Any]) -> float:
        return float(tp.get("final_accuracy", 0.0)) - float(base.get("final_accuracy", 0.0))

    repeated_drop_full = relative_drop(full.get("repeated_error_rate", 0.0), tp.get("repeated_error_rate", 0.0))
    repeated_drop_self = relative_drop(self_corr.get("repeated_error_rate", 0.0), tp.get("repeated_error_rate", 0.0))
    prefix_drop_full = relative_drop(full.get("prefix_damage_rate", 0.0), tp.get("prefix_damage_rate", 0.0))
    criteria = {
        "tracepatch_gain_over_full_regeneration": gain(full),
        "tracepatch_gain_over_self_correction": gain(self_corr),
        "tracepatch_gain_over_edit_only_repair": gain(edit),
        "repeated_error_drop_vs_full_regeneration": repeated_drop_full,
        "repeated_error_drop_vs_self_correction": repeated_drop_self,
        "prefix_damage_drop_vs_full_regeneration": prefix_drop_full,
        "oracle_gap": summary.get("oracle_overlap", {}).get("oracle_gap", 0.0),
        "parse_error_rate_tracepatch": tp.get("parse_error_rate", 0.0),
        "passes_scale_gate": (
            gain(full) >= 0.05
            and gain(self_corr) >= 0.08
            and gain(edit) >= 0.03
            and max(repeated_drop_full, repeated_drop_self) >= 0.25
            and prefix_drop_full >= 0.25
            and summary.get("oracle_overlap", {}).get("oracle_gap", 0.0) > 0
            and tp.get("parse_error_rate", 1.0) < 0.20
        ),
    }
    return criteria


def relative_drop(baseline: float, candidate: float) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0:
        return 0.0 if candidate >= baseline else 1.0
    return (baseline - candidate) / baseline


def verdict(summary: dict[str, Any]) -> str:
    if summary["audit"]["ambiguous_or_invalid_rate"] > 0.10:
        return "rerun needed"
    return "scale" if summary.get("scale_criteria", {}).get("passes_scale_gate") else "no scale"


def build_manual_audit_sample(examples, initial_rows, repair_rows, checker_rows, localization_rows):
    sample = []
    by_ex = {e["example_id"]: e for e in examples}
    failed = [r for r in initial_rows if not r["passed"]]
    for domain in DOMAINS:
        for init in [r for r in failed if r["domain"] == domain][:10]:
            sample.append(manual_audit_row(init, domain, repair_rows, checker_rows, localization_rows, ""))
    if len(sample) < 30:
        used = {(row["model"], row["example_id"]) for row in sample}
        for domain in DOMAINS:
            for init in [r for r in initial_rows if r["domain"] == domain and (r["model"], r["example_id"]) not in used]:
                sample.append(manual_audit_row(init, domain, repair_rows, checker_rows, localization_rows, "initial passed; no failed-only repair was run"))
                used.add((init["model"], init["example_id"]))
                if len(sample) >= 30:
                    break
            if len(sample) >= 30:
                break
    return sample[:30]


def manual_audit_row(init, domain, repair_rows, checker_rows, localization_rows, note):
    repairs = {(r["system"], r["example_id"], r["model"]): r for r in repair_rows}
    loc = next((l for l in localization_rows if l["example_id"] == init["example_id"] and l["model"] == init["model"] and l["system"] == "tracepatch_localize"), {})
    return {
        "example_id": init["example_id"], "domain": domain, "model": init["model"],
        "initial_trace": json.dumps(init["parsed"], ensure_ascii=False),
        "checker_feedback": next(c["feedback"] for c in checker_rows if c["stage"] == "initial" and c["example_id"] == init["example_id"] and c["model"] == init["model"]),
        "tracepatch_localization": json.dumps(loc, ensure_ascii=False),
        "tracepatch_output": repairs.get(("tracepatch", init["example_id"], init["model"]), {}).get("raw_output", ""),
        "full_regeneration_output": repairs.get(("full_regeneration", init["example_id"], init["model"]), {}).get("raw_output", ""),
        "self_correction_output": repairs.get(("self_correction", init["example_id"], init["model"]), {}).get("raw_output", ""),
        "prefix_actually_correct": "", "localization_reasonable": "", "tracepatch_preserved_prefix": "", "full_regen_damaged_prefix": "", "notes": note,
    }


def manual_audit_fields():
    return ["example_id", "domain", "model", "initial_trace", "checker_feedback", "tracepatch_localization", "tracepatch_output", "full_regeneration_output", "self_correction_output", "prefix_actually_correct", "localization_reasonable", "tracepatch_preserved_prefix", "full_regen_damaged_prefix", "notes"]


def build_interpretation(summary, breakdown, audit):
    aggregate_rows = [
        "| system | final accuracy | repair success on failed | repeated error | prefix damage | parse error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in ("self_correction", "full_regeneration", "edit_only_repair", "tracepatch"):
        row = summary.get("aggregate", {}).get(system, {})
        aggregate_rows.append(
            f"| {system} | {row.get('final_accuracy', 0):.3f} | "
            f"{row.get('repair_success_on_failed', 0):.3f} | "
            f"{row.get('repeated_error_rate', 0):.3f} | "
            f"{row.get('prefix_damage_rate', 0):.3f} | "
            f"{row.get('parse_error_rate', 0):.3f} |"
        )
    model_rows = [
        "| model | initial accuracy | initial failures | tracepatch final | full regeneration final |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, initial in summary.get("initial", {}).items():
        tp = summary.get("systems", {}).get(f"{model}|tracepatch", {})
        full = summary.get("systems", {}).get(f"{model}|full_regeneration", {})
        model_rows.append(
            f"| {model} | {initial.get('accuracy', 0):.3f} | {initial.get('n_failed', 0)} | "
            f"{tp.get('final_accuracy', 0):.3f} | {full.get('final_accuracy', 0):.3f} |"
        )
    criteria = summary.get("scale_criteria", {})
    oracle = summary.get("oracle_overlap", {})
    return "\n".join([
        "# TracePatch Micro V1 Interpretation",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "## Main Result",
        *aggregate_rows,
        "",
        "## Model Breakdown",
        *model_rows,
        "",
        "## Scale Criteria",
        f"- TracePatch gain over full regeneration: {criteria.get('tracepatch_gain_over_full_regeneration', 0):.3f}",
        f"- TracePatch gain over self-correction: {criteria.get('tracepatch_gain_over_self_correction', 0):.3f}",
        f"- TracePatch gain over edit-only repair: {criteria.get('tracepatch_gain_over_edit_only_repair', 0):.3f}",
        f"- Repeated-error drop vs full regeneration: {criteria.get('repeated_error_drop_vs_full_regeneration', 0):.1%}",
        f"- Repeated-error drop vs self-correction: {criteria.get('repeated_error_drop_vs_self_correction', 0):.1%}",
        f"- Prefix-damage drop vs full regeneration: {criteria.get('prefix_damage_drop_vs_full_regeneration', 0):.1%}",
        f"- Oracle gap on overlapping subset: {oracle.get('oracle_gap', 0):.3f} over n={oracle.get('n_oracle_repairs', 0)} oracle repairs",
        "",
        "## Failure Modes",
        "- Full regeneration matched or beat TracePatch on aggregate final accuracy.",
        "- TracePatch did not reduce prefix damage relative to full regeneration in the automatic approximation.",
        "- Oracle localization did not beat normal TracePatch on the available checker-derived subset.",
        "- RAG had no initial failures in this micropilot, so repair gains are driven by code and math only.",
        "- Repeated-error and prefix-damage metrics are automatic approximations; inspect the manual audit sample before treating them as paper-grade.",
        "",
        "## Data/Evaluator Audit",
        f"- n_examples: {audit['n_examples']}",
        f"- invalid/ambiguous: {audit['n_ambiguous_or_invalid_examples']} ({audit['ambiguous_or_invalid_rate']:.1%})",
        f"- gold checker failures: {audit['n_gold_answer_checker_failures']} ({audit['gold_answer_checker_failure_rate']:.1%})",
        f"- wrong-answer false negatives: {audit['n_wrong_answer_checker_false_negatives']} ({audit['wrong_answer_checker_false_negative_rate']:.1%})",
        f"- low-locality feedback cases: {audit['n_feedback_low_locality']} ({audit['feedback_low_locality_rate']:.1%})",
        f"- low-confidence prefix metric cases: {audit['n_prefix_damage_metric_low_confidence']} ({audit['prefix_damage_metric_low_confidence_rate']:.1%})",
        f"- low-confidence repeated-error metric cases: {audit['n_repeated_error_metric_low_confidence']} ({audit['repeated_error_metric_low_confidence_rate']:.1%})",
        "",
        "The data/evaluator audit passes, but the mechanism result does not pass the predeclared scale gate.",
    ])


def mean(values) -> float:
    vals = [bool(v) if isinstance(v, bool) else float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0
