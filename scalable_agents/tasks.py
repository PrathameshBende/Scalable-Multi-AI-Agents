"""30 multi-step agent tasks with automatically computed expected answers.

Each task's ``solve`` uses the same tool functions the agent has, so the
expected answer cannot drift from the data. ``min_tools`` is the number of
distinct tool calls a correct solution needs (arithmetic counts as one).

Grading (``grade``): the agent must end with a line ``FINAL ANSWER: <number>``.
- strict  : that number is within 1% (or 0.05 absolute) of the expected value.
- lenient : ANY number in the reply is within tolerance (reported, not headline).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from .tools import (
    convert_currency as cc,
    customer_discount as disc,
    days_between as days,
    get_warehouse as wh,
    lookup_product as p,
    shipping_cost as ship,
)


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    question: str
    solve: Callable[[], float]
    min_tools: int

    @property
    def expected(self) -> float:
        return float(self.solve())


def _price(name, qty=1):
    return p(name)["price_usd"] * qty


def _weight(name, qty=1):
    return p(name)["weight_kg"] * qty


def _fee(product):
    return wh(p(product)["warehouse"])["storage_fee_usd_per_unit_per_day"]


TASKS: list[Task] = [
    # A. product lookup + arithmetic
    Task("A01", "lookup", "What is the total price in USD of 3 Aurora Desk Lamps?",
         lambda: _price("aurora desk lamp", 3), 2),
    Task("A02", "lookup", "What is the total weight in kg of 12 Quartz Water Bottles?",
         lambda: _weight("quartz water bottle", 12), 2),
    Task("A03", "lookup", "A customer buys 45 Atlas Backpacks. How many Atlas Backpacks remain in stock?",
         lambda: p("atlas backpack")["stock"] - 45, 2),
    Task("A04", "lookup", "What is the total price in USD of 2 Orion Chairs and 1 Lumen Monitor Arm?",
         lambda: _price("orion chair", 2) + _price("lumen monitor arm"), 3),
    Task("A05", "lookup", "What would the revenue in USD be if every Vega Headphones unit in stock were sold at its list price?",
         lambda: p("vega headphones")["stock"] * _price("vega headphones"), 2),
    Task("A06", "lookup", "How much more expensive, in USD, is an Orion Chair than a Zephyr Fan?",
         lambda: _price("orion chair") - _price("zephyr fan"), 3),
    Task("A07", "lookup", "How many whole Nimbus Kettles can be bought with 500 USD?",
         lambda: math.floor(500 / _price("nimbus kettle")), 2),
    Task("A08", "lookup", "What is the average price in USD of an Aurora Desk Lamp, a Nimbus Kettle and a Zephyr Fan?",
         lambda: (_price("aurora desk lamp") + _price("nimbus kettle") + _price("zephyr fan")) / 3, 4),
    Task("A09", "lookup", "What is the combined stock of Zephyr Fans and Orion Chairs?",
         lambda: p("zephyr fan")["stock"] + p("orion chair")["stock"], 3),
    Task("A10", "lookup", "What is the price per kg, in USD, of a Lumen Monitor Arm?",
         lambda: _price("lumen monitor arm") / _weight("lumen monitor arm"), 2),
    # B. shipping
    Task("B01", "shipping", "How much does it cost in USD to ship 4 Nimbus Kettles to the south region?",
         lambda: ship("south", _weight("nimbus kettle", 4)), 3),
    Task("B02", "shipping", "How much does it cost in USD to ship one Orion Chair internationally?",
         lambda: ship("international", _weight("orion chair")), 2),
    Task("B03", "shipping", "What is the total cost in USD (item price plus shipping) of 2 Zephyr Fans shipped to the east region?",
         lambda: _price("zephyr fan", 2) + ship("east", _weight("zephyr fan", 2)), 4),
    Task("B04", "shipping", "How much does it cost in USD to ship 10 Atlas Backpacks to the north region?",
         lambda: ship("north", _weight("atlas backpack", 10)), 3),
    Task("B05", "shipping", "One Aurora Desk Lamp is shipped to the region where its own warehouse is located. What is the shipping cost in USD?",
         lambda: ship(wh(p("aurora desk lamp")["warehouse"])["region"], _weight("aurora desk lamp")), 3),
    Task("B06", "shipping", "What is the total cost in USD (prices plus shipping) of 5 Quartz Water Bottles and 1 Vega Headphones shipped together to the west region?",
         lambda: _price("quartz water bottle", 5) + _price("vega headphones")
         + ship("west", _weight("quartz water bottle", 5) + _weight("vega headphones")), 5),
    # C. currency
    Task("C01", "currency", "What is the price of one Zephyr Fan in INR?",
         lambda: cc(_price("zephyr fan"), "USD", "INR"), 2),
    Task("C02", "currency", "What is the total price of 3 Vega Headphones in EUR?",
         lambda: cc(_price("vega headphones", 3), "USD", "EUR"), 3),
    Task("C03", "currency", "A customer has 10000 INR. Converted to USD, how many whole Quartz Water Bottles can they buy?",
         lambda: math.floor(cc(10000, "INR", "USD") / _price("quartz water bottle")), 3),
    Task("C04", "currency", "What is the price of one Orion Chair in GBP?",
         lambda: cc(_price("orion chair"), "USD", "GBP"), 2),
    Task("C05", "currency", "What is the cost in JPY of shipping 2 Lumen Monitor Arms to the east region?",
         lambda: cc(ship("east", _weight("lumen monitor arm", 2)), "USD", "JPY"), 4),
    Task("C06", "currency", "A customer pays 5000 INR for one Nimbus Kettle. How much change do they get, in INR?",
         lambda: 5000 - cc(_price("nimbus kettle"), "USD", "INR"), 3),
    # D. discounts, dates, warehouses (longest chains)
    Task("D01", "mixed", "A gold-tier customer buys 2 Orion Chairs. What do they pay in USD after their tier discount?",
         lambda: _price("orion chair", 2) * (1 - disc("gold")), 3),
    Task("D02", "mixed", "A silver-tier customer buys 4 Atlas Backpacks shipped to the north region. The discount applies to items only, not shipping. What is the total in USD?",
         lambda: _price("atlas backpack", 4) * (1 - disc("silver")) + ship("north", _weight("atlas backpack", 4)), 5),
    Task("D03", "mixed", "All 22 Orion Chairs sit in their warehouse from 2025-06-01 to 2025-06-30. Using that warehouse's per-unit daily storage fee and the number of days between those dates, what is the total storage cost in USD?",
         lambda: 22 * days("2025-06-01", "2025-06-30") * _fee("orion chair"), 4),
    Task("D04", "mixed", "An order of 10 Zephyr Fans was placed on 2025-05-14 and delivered on 2025-07-02. The Zephyr Fan warehouse's per-unit daily storage fee is charged for every day between those dates. What is the total fee in USD?",
         lambda: 10 * days("2025-05-14", "2025-07-02") * _fee("zephyr fan"), 4),
    Task("D05", "mixed", "What is the combined weight in kg of 5 Vega Headphones and 4 Quartz Water Bottles?",
         lambda: _weight("vega headphones", 5) + _weight("quartz water bottle", 4), 3),
    Task("D06", "mixed", "A gold-tier customer buys 1 Lumen Monitor Arm and pays in EUR. How much do they pay in EUR after the discount?",
         lambda: cc(_price("lumen monitor arm") * (1 - disc("gold")), "USD", "EUR"), 4),
    Task("D07", "mixed", "A bronze-tier customer buys 3 Aurora Desk Lamps shipped internationally. What is the total (items after discount plus shipping) in GBP?",
         lambda: cc(_price("aurora desk lamp", 3) * (1 - disc("bronze"))
                    + ship("international", _weight("aurora desk lamp", 3)), "USD", "GBP"), 5),
    Task("D08", "mixed", "What does it cost in USD to store every Quartz Water Bottle currently in stock for 15 days at its warehouse's per-unit daily storage fee?",
         lambda: p("quartz water bottle")["stock"] * 15 * _fee("quartz water bottle"), 3),
]

_FINAL = re.compile(r"final answer\s*[:=]\s*(.+)", re.IGNORECASE)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def parse_final_answer(text: str) -> float | None:
    """Number on the last 'FINAL ANSWER:' line, or None if absent."""
    matches = _FINAL.findall(text or "")
    if not matches:
        return None
    nums = _NUM.findall(matches[-1])
    return _to_float(nums[0]) if nums else None


def all_numbers(text: str) -> list[float]:
    return [v for v in (_to_float(n) for n in _NUM.findall(text or "")) if v is not None]


def close(pred: float | None, expected: float, rel: float = 0.01, abs_tol: float = 0.05) -> bool:
    return pred is not None and abs(pred - expected) <= max(abs_tol, rel * abs(expected))


def grade(task: Task, reply: str) -> dict:
    exp = task.expected
    final = parse_final_answer(reply)
    return {
        "expected": exp,
        "parsed": final,
        "strict": close(final, exp),
        "lenient": any(close(v, exp) for v in all_numbers(reply)),
    }
