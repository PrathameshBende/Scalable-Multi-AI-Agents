import math

import pytest

from scalable_agents import tools as T
from scalable_agents.tasks import TASKS, all_numbers, close, grade, parse_final_answer


def test_tools_basic():
    assert T.lookup_product("Zephyr Fans")["price_usd"] == 49.0
    assert T.get_warehouse("Pune")["region"] == "west"
    assert T.shipping_cost("East", 6.8) == 35.7
    assert T.convert_currency(49, "usd", "INR") == 4067.0
    assert T.customer_discount("gold") == 0.12
    assert T.days_between("2025-06-01", "2025-06-30") == 29


@pytest.mark.parametrize("bad", [
    lambda: T.lookup_product("toaster"), lambda: T.shipping_cost("mars", 1),
    lambda: T.convert_currency(1, "USD", "BTC"), lambda: T.days_between("June 1", "2025-06-30"),
    lambda: T.calculator("__import__('os')"), lambda: T.calculator("1/0"),
    lambda: T.calculator("2**1000"),
])
def test_tool_errors_are_tool_errors(bad):
    with pytest.raises(T.ToolError):
        bad()


def test_calculator():
    assert T.calculator("3 * 34.5 + 2.75") == pytest.approx(106.25)
    assert T.calculator("floor(500 / 27.99)") == 17
    assert T.calculator("1,000 * 2") == 2000
    assert T.calculator("2^3") == 8


def test_thirty_unique_multistep_tasks():
    assert len(TASKS) == 30
    assert len({t.id for t in TASKS}) == 30
    assert all(t.min_tools >= 2 for t in TASKS)
    assert all(math.isfinite(t.expected) for t in TASKS)


def test_dev_set_is_separate_from_test_set():
    from scalable_agents.tasks import DEV_TASKS

    assert len(DEV_TASKS) == 12 and all(math.isfinite(t.expected) for t in DEV_TASKS)
    assert not {t.question for t in DEV_TASKS} & {t.question for t in TASKS}


# Expected answers worked out by hand from the data tables, as a cross-check.
HAND = {
    "A01": 103.5, "A07": 17, "A08": 37.163, "B03": 133.7, "B05": 3.3, "B06": 157.48,
    "C03": 9, "C05": 6500.55, "C06": 2676.83, "D02": 241.45, "D03": 31.9, "D04": 14.7,
    "D07": 115.89, "D08": 270.0,
}


@pytest.mark.parametrize("tid,value", HAND.items())
def test_expected_matches_hand_calculation(tid, value):
    task = next(t for t in TASKS if t.id == tid)
    assert task.expected == pytest.approx(value, abs=0.01)


def test_grading():
    task = TASKS[0]  # expected 103.5
    assert grade(task, "3 x 34.50 = 103.50\nFINAL ANSWER: 103.50")["strict"]
    g = grade(task, "The total is $103.50 (3 x $34.50).")
    assert not g["strict"] and g["lenient"]  # no FINAL ANSWER line -> strict fail
    assert not grade(task, "FINAL ANSWER: 100")["strict"]
    assert parse_final_answer("final answer = $1,939.05") == 1939.05
    assert parse_final_answer("no answer") is None
    assert all_numbers("a -2 and 3.5") == [-2.0, 3.5]
    assert close(100.9, 100) and not close(102, 100)
