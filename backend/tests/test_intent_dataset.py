import json
import logging
from pathlib import Path
from typing import Any

import pytest

from app.assistant.planner import Planner
from app.assistant.request_classifier import RequestCategory, RequestClassification, classify_request

LOGGER = logging.getLogger(__name__)
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "intent_cases.jsonl"


def test_intent_dataset_classifier_and_planner_outputs() -> None:
    cases = _load_cases()
    planner = Planner()
    failures: list[str] = []
    passed_cases = 0
    false_tool_execution_count = 0
    missed_tool_execution_count = 0
    wrong_category_count = 0

    for index, case in enumerate(cases, start=1):
        message = str(case["message"])
        classification = classify_request(message, planner)
        planner_decision = planner.decide_action(message)
        case_failures: list[str] = []

        actual_category = classification.category.value
        expected_category = case["expected_category"]
        if actual_category != expected_category:
            wrong_category_count += 1
            case_failures.append(f"category expected {expected_category!r}, got {actual_category!r}")

        actual_intent_type = classification.action_decision.intent_type if classification.action_decision else None
        expected_intent_type = case["expected_intent_type"]
        if actual_intent_type != expected_intent_type:
            case_failures.append(f"intent expected {expected_intent_type!r}, got {actual_intent_type!r}")

        if expected_intent_type is not None and planner_decision.intent_type != expected_intent_type:
            case_failures.append(f"planner intent expected {expected_intent_type!r}, got {planner_decision.intent_type!r}")

        actual_tool = _actual_tool_name(classification)
        expected_tool = case["expected_tool"]
        if actual_tool != expected_tool:
            case_failures.append(f"tool expected {expected_tool!r}, got {actual_tool!r}")

        should_execute_tool = bool(case["should_execute_tool"])
        actual_should_execute_tool = _should_execute_tool(classification)
        if actual_should_execute_tool and not should_execute_tool:
            false_tool_execution_count += 1
            case_failures.append("classifier would execute a tool but fixture expected no execution")
        if should_execute_tool and not actual_should_execute_tool:
            missed_tool_execution_count += 1
            case_failures.append("classifier missed expected tool execution")

        _assert_planner_tool_output(case, planner, case_failures)

        if case_failures:
            failures.append(f"case {index} message={message!r}: " + "; ".join(case_failures))
        else:
            passed_cases += 1

    metrics = {
        "total_cases": len(cases),
        "passed_cases": passed_cases,
        "failed_cases": len(failures),
        "false_tool_execution_count": false_tool_execution_count,
        "missed_tool_execution_count": missed_tool_execution_count,
        "wrong_category_count": wrong_category_count,
    }
    LOGGER.info("intent dataset metrics: %s", metrics)
    print(f"intent dataset metrics: {json.dumps(metrics, sort_keys=True)}")

    assert false_tool_execution_count == 0
    assert not failures, "\n".join(failures)


def _load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with FIXTURE_PATH.open(encoding="utf-8") as fixture:
        for line_number, line in enumerate(fixture, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            _validate_case_shape(case, line_number)
            cases.append(case)

    assert len(cases) >= 120
    return cases


def _validate_case_shape(case: dict[str, Any], line_number: int) -> None:
    required_keys = {
        "message",
        "expected_category",
        "expected_intent_type",
        "expected_tool",
        "should_execute_tool",
    }
    optional_keys = {"notes"}
    assert required_keys <= set(case) <= required_keys | optional_keys, f"line {line_number} has invalid keys: {sorted(case)}"
    assert isinstance(case["message"], str), f"line {line_number} message must be a string"
    assert case["expected_category"] in {category.value for category in RequestCategory}, (
        f"line {line_number} has invalid expected_category"
    )
    assert case["expected_intent_type"] in {
        None,
        "EXECUTE_TOOL",
        "ANSWER_ONLY",
        "NO_ACTION",
        "ASK_CLARIFICATION",
        "MISSING_CAPABILITY",
        "BLOCKED",
    }, f"line {line_number} has invalid expected_intent_type"
    assert case["expected_tool"] is None or isinstance(case["expected_tool"], str), (
        f"line {line_number} expected_tool must be null or string"
    )
    assert isinstance(case["should_execute_tool"], bool), f"line {line_number} should_execute_tool must be boolean"


def _actual_tool_name(classification: RequestClassification) -> str | None:
    if classification.tool_sequence:
        return ",".join(proposal.name for proposal in classification.tool_sequence)
    if classification.tool_proposal is not None:
        return classification.tool_proposal.name
    return None


def _should_execute_tool(classification: RequestClassification) -> bool:
    if classification.category == RequestCategory.multi_step:
        return bool(classification.tool_sequence)
    if classification.category in {RequestCategory.local_tool, RequestCategory.local_context}:
        return classification.tool_proposal is not None
    if classification.category == RequestCategory.realtime_info:
        return classification.tool_proposal is not None
    return False


def _assert_planner_tool_output(case: dict[str, Any], planner: Planner, failures: list[str]) -> None:
    message = str(case["message"])
    expected_category = case["expected_category"]
    expected_tool = case["expected_tool"]

    if expected_category == RequestCategory.local_tool.value:
        proposal = planner.propose_tool_call(message)
        if proposal is None:
            failures.append("planner did not return expected single-tool proposal")
        elif proposal.name != expected_tool:
            failures.append(f"planner tool expected {expected_tool!r}, got {proposal.name!r}")

    if expected_category == RequestCategory.multi_step.value:
        sequence = planner.propose_tool_sequence(message)
        actual_sequence = ",".join(proposal.name for proposal in sequence) if sequence else None
        if actual_sequence != expected_tool:
            failures.append(f"planner sequence expected {expected_tool!r}, got {actual_sequence!r}")

    if case["should_execute_tool"] is False:
        proposal = planner.propose_tool_call(message)
        sequence = planner.propose_tool_sequence(message)
        if proposal is not None or sequence:
            failures.append("planner proposed a deterministic tool for a no-execution case")
