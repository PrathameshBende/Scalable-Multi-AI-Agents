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


def test_fresh_tasks_are_new_balanced_and_reproducible():
    from scalable_agents.tasks import DEV_TASKS, FRESH_TASKS, make_fresh_tasks

    assert len(FRESH_TASKS) == 30
    cats = [t.category for t in FRESH_TASKS]
    assert [cats.count(c) for c in ("lookup", "shipping", "currency", "mixed")] == [8, 8, 7, 7]
    old = {t.question for t in TASKS} | {t.question for t in DEV_TASKS}
    assert not old & {t.question for t in FRESH_TASKS}
    assert [t.question for t in make_fresh_tasks()] == [t.question for t in FRESH_TASKS]
    assert all(math.isfinite(t.expected) for t in FRESH_TASKS)


def test_ship_products_does_the_weight_math():
    # 4 Nimbus Kettles (1.8 kg each) to the south (3.5 USD/kg) = 7.2 kg * 3.5 = 25.2
    assert T.ship_products("south", "Nimbus Kettle", 4) == T.shipping_cost("south", 7.2) == 25.2
    with pytest.raises(T.ToolError):
        T.ship_products("south", "nimbus kettle", 2.5)
    with pytest.raises(T.ToolError):
        T.ship_products("mars", "nimbus kettle", 1)


def test_task_answers_reachable_with_per_product_shipping():
    # B06 ships two products together; with the v2 tool the agent adds two per-product
    # quotes. Rounding each quote separately must stay within grading tolerance.
    b06 = next(t for t in TASKS if t.id == "B06")
    via_v2 = (T.lookup_product("quartz water bottle")["price_usd"] * 5 + T.lookup_product("vega headphones")["price_usd"]
              + T.ship_products("west", "quartz water bottle", 5) + T.ship_products("west", "vega headphones", 1))
    assert close(via_v2, b06.expected)


def test_majority_vote():
    from scalable_agents.tasks import majority_vote

    assert majority_vote([5.0]) == 0
    assert majority_vote([5.0, 7.0, 7.0]) == 1          # majority beats the greedy run
    assert majority_vote([5.0, 7.0, 9.0]) == 0          # no agreement: greedy run
    assert majority_vote([None, 7.0, 7.004]) == 1       # within tolerance counts as agreement
    assert majority_vote([None, None, None]) == 0


def test_round2_task_sets_are_new_balanced_and_disjoint():
    from scalable_agents.tasks import DEV2_TASKS, DEV_TASKS, FRESH2_TASKS, FRESH_TASKS

    sets = {"test": TASKS, "dev": DEV_TASKS, "fresh": FRESH_TASKS, "fresh2": FRESH2_TASKS, "dev2": DEV2_TASKS}
    qs = {k: {t.question for t in v} for k, v in sets.items()}
    names = list(qs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not qs[a] & qs[b], (a, b)
    for ts, prefix in ((FRESH2_TASKS, "G"), (DEV2_TASKS, "V")):
        cats = [t.category for t in ts]
        assert [cats.count(c) for c in ("lookup", "shipping", "currency", "mixed")] == [8, 8, 7, 7]
        assert all(t.id.startswith(prefix) and math.isfinite(t.expected) for t in ts)
