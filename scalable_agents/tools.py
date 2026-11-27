"""Pure-Python tools for the ReAct agent, plus their (fictional) data.

The data is invented on purpose: the model cannot answer from memory, so a
correct answer requires actually calling the tools. Functions here return plain
Python values; ``agent.py`` wraps them as LangChain tools that return JSON text.
"""

from __future__ import annotations

import ast
import datetime as _dt
import math
import operator

PRODUCTS: dict[str, dict] = {
    "aurora desk lamp": {"price_usd": 34.50, "weight_kg": 1.2, "stock": 140, "warehouse": "Pune"},
    "nimbus kettle": {"price_usd": 27.99, "weight_kg": 1.8, "stock": 85, "warehouse": "Chennai"},
    "zephyr fan": {"price_usd": 49.00, "weight_kg": 3.4, "stock": 60, "warehouse": "Kolkata"},
    "orion chair": {"price_usd": 129.00, "weight_kg": 9.5, "stock": 22, "warehouse": "Pune"},
    "atlas backpack": {"price_usd": 59.75, "weight_kg": 0.9, "stock": 310, "warehouse": "Delhi"},
    "lumen monitor arm": {"price_usd": 74.20, "weight_kg": 4.1, "stock": 47, "warehouse": "Chennai"},
    "quartz water bottle": {"price_usd": 12.40, "weight_kg": 0.35, "stock": 900, "warehouse": "Delhi"},
    "vega headphones": {"price_usd": 89.90, "weight_kg": 0.28, "stock": 133, "warehouse": "Kolkata"},
}

WAREHOUSES: dict[str, dict] = {
    "pune": {"region": "west", "storage_fee_usd_per_unit_per_day": 0.05},
    "chennai": {"region": "south", "storage_fee_usd_per_unit_per_day": 0.04},
    "kolkata": {"region": "east", "storage_fee_usd_per_unit_per_day": 0.03},
    "delhi": {"region": "north", "storage_fee_usd_per_unit_per_day": 0.02},
}

SHIPPING_USD_PER_KG = {"north": 4.0, "south": 3.5, "east": 5.25, "west": 2.75, "international": 12.0}

# Units of each currency per 1 USD (fixed company rates, not live rates).
PER_USD = {"USD": 1.0, "INR": 83.0, "EUR": 0.92, "GBP": 0.79, "JPY": 151.0}

TIER_DISCOUNT = {"bronze": 0.0, "silver": 0.05, "gold": 0.12}


class ToolError(ValueError):
    """Raised for bad tool input; the agent sees the message and can retry."""


def _key(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").split())


def lookup_product(name: str) -> dict:
    key = _key(name)
    if key.endswith("s") and key not in PRODUCTS and key[:-1] in PRODUCTS:
        key = key[:-1]  # "zephyr fans" -> "zephyr fan"
    if key not in PRODUCTS:
        raise ToolError(f"unknown product {name!r}; known: {sorted(PRODUCTS)}")
    return {"name": key, **PRODUCTS[key]}


def get_warehouse(name: str) -> dict:
    key = _key(name)
    if key not in WAREHOUSES:
        raise ToolError(f"unknown warehouse {name!r}; known: {sorted(WAREHOUSES)}")
    return {"warehouse": key, **WAREHOUSES[key]}


def shipping_cost(region: str, weight_kg: float) -> float:
    key = _key(region)
    if key not in SHIPPING_USD_PER_KG:
        raise ToolError(f"unknown region {region!r}; known: {sorted(SHIPPING_USD_PER_KG)}")
    if weight_kg < 0:
        raise ToolError("weight must be non-negative")
    return round(SHIPPING_USD_PER_KG[key] * float(weight_kg), 2)


def ship_products(region: str, product: str, quantity: int = 1) -> float:
    """Shipping cost for `quantity` units of one product: the tool does the weight lookup
    and multiplication itself, so the model cannot mix up unit and total weight."""
    if quantity < 0 or float(quantity) != int(quantity):
        raise ToolError("quantity must be a non-negative whole number")
    return shipping_cost(region, lookup_product(product)["weight_kg"] * int(quantity))


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    f, t = from_currency.upper().strip(), to_currency.upper().strip()
    for c in (f, t):
        if c not in PER_USD:
            raise ToolError(f"unknown currency {c!r}; known: {sorted(PER_USD)}")
    return round(float(amount) / PER_USD[f] * PER_USD[t], 2)


def customer_discount(tier: str) -> float:
    key = _key(tier)
    if key not in TIER_DISCOUNT:
        raise ToolError(f"unknown tier {tier!r}; known: {sorted(TIER_DISCOUNT)}")
    return TIER_DISCOUNT[key]


def days_between(start_date: str, end_date: str) -> int:
    try:
        a = _dt.date.fromisoformat(start_date.strip())
        b = _dt.date.fromisoformat(end_date.strip())
    except ValueError as e:
        raise ToolError(f"dates must be YYYY-MM-DD: {e}") from None
    return (b - a).days


# --- safe calculator --------------------------------------------------------
_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "round": round, "floor": math.floor, "ceil": math.ceil, "abs": abs,
    "min": min, "max": max, "sqrt": math.sqrt,
}


def calculator(expression: str) -> float:
    """Evaluate arithmetic without eval(): only numbers, + - * / // % **, and _FUNCS."""
    expr = expression.replace(",", "").replace("^", "**").replace("×", "*")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords
        ):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ToolError(f"unsupported expression: {ast.dump(node)[:80]}")

    try:
        tree = ast.parse(expr, mode="eval")
        result = ev(tree)
    except ToolError:
        raise
    except ZeroDivisionError:
        raise ToolError("division by zero") from None
    except (SyntaxError, TypeError, ValueError) as e:
        raise ToolError(f"could not evaluate {expression!r}: {e}") from None
    return float(result)
