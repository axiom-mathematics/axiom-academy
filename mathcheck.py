"""Exact mathematical answer checking.

SymPy decides correctness. An LLM is never consulted. The same module is
copied to ``student-pages/mathcheck.py`` and executed in the browser with
Pyodide, so keep this file free of application imports.

A numeric answer may be exact, or a decimal that is accurate to three places
after the decimal point, rounded or truncated. That is the default. A problem
can require exact form only, or a different number of places.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, localcontext
import re
from typing import Any

import sympy as sp
from sympy import FiniteSet, Interval, Tuple, Union
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass
class CheckResult:
    status: str
    message: str
    error_category: str | None = None
    equivalent: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "error_category": self.error_category,
            "equivalent": self.equivalent,
        }


def normalize_decimal_places(value: int | None) -> int | None:
    """``None`` and omitted both mean three places. A negative value means exact only."""
    if value is None:
        return 3
    number = int(value)
    if number < 0:
        return None
    return number


def answer_note(decimal_places: int | None = 3) -> str:
    """The line shown under a student's answer box."""
    places = normalize_decimal_places(decimal_places)
    if places is None:
        return "Enter an exact answer. A decimal approximation is not accepted for this problem."
    if places == 3:
        return "Enter an exact answer (like 7/3) or a decimal to three places (like 2.333)."
    word = "place" if places == 1 else "places"
    return f"Enter an exact answer or a decimal accurate to {places} {word}."


def decimal_preview(answer: str, *, kind: str = "auto", decimal_places: int | None = 3) -> str | None:
    """Three-place (or N-place) form of a numeric key, for the instructor.

    Returns ``None`` when the key is not a number, point, vector, interval, or set of numbers,
    and when the problem is exact-only.
    """
    places = normalize_decimal_places(decimal_places)
    if places is None or not str(answer or "").strip():
        return None
    try:
        text = preprocess(answer)
        tagged = interpret(text, kind)
    except ValueError:
        return None
    rendered = _preview_value(text, tagged, places)
    return rendered or None


def check_answer(
    expected: str,
    given: str,
    *,
    tolerance: float = 1e-3,
    kind: str = "auto",
    decimal_places: int | None = 3,
) -> CheckResult:
    """Compare ``given`` with ``expected``.

    ``status`` is ``correct``, ``incorrect``, or ``invalid``. Malformed input
    is ``invalid`` and is not treated as a wrong answer.
    """
    if given is None or not str(given).strip():
        return CheckResult("invalid", "Enter an answer before checking.", equivalent=None)
    if expected is None or not str(expected).strip():
        return CheckResult("invalid", "This problem has no answer key yet.", equivalent=None)

    try:
        exp_value = interpret(str(expected), kind)
        got_value = interpret(str(given), kind if kind != "auto" else _kind_for_pair(expected, given, kind))
    except ValueError as exc:
        return CheckResult("invalid", str(exc), equivalent=None)

    # A solution list and a set of the same numbers are the same answer.
    exp_value, got_value = _align_solutions(exp_value, got_value)

    if exp_value[0] != got_value[0]:
        # x > 2 and (2, oo) describe the same solution set.
        converted = _coerce_pair(exp_value, got_value)
        if converted is None:
            return CheckResult(
                "incorrect",
                "That doesn't match the kind of answer this problem expects.",
                error_category="form_mismatch",
                equivalent=False,
            )
        exp_value, got_value = converted

    places = normalize_decimal_places(decimal_places)
    notes: list[str] = []
    try:
        same = _equivalent(
            exp_value,
            got_value,
            tolerance,
            places,
            preprocess(str(expected)),
            preprocess(str(given)),
            notes,
        )
    except ValueError as exc:
        return CheckResult("invalid", str(exc), equivalent=None)

    if same:
        return CheckResult("correct", "That's equivalent.", error_category=None, equivalent=True)

    if notes:
        return CheckResult("incorrect", notes[0], error_category="decimal_places", equivalent=False)

    category = _categorize(exp_value, got_value, tolerance)
    return CheckResult(
        "incorrect",
        _student_message(category),
        error_category=category,
        equivalent=False,
    )


def interpret(text: str, kind: str = "auto") -> tuple[str, Any]:
    """Parse math text into a tagged SymPy value."""
    cleaned = preprocess(text)
    if not cleaned:
        raise ValueError("Enter an answer before checking.")
    mode = (kind or "auto").strip().lower()
    if mode in {"vector", "point"}:
        return ("vector", _parse_vector(cleaned))
    if mode == "interval":
        return ("interval", _parse_interval(cleaned))
    if mode == "set":
        return ("set", _parse_set_or_solutions(cleaned))
    if mode == "equation":
        return _parse_equation_or_relation(cleaned)
    if mode not in {"auto", "expression", "number"}:
        raise ValueError(f"Unknown answer kind {kind!r}.")

    if _is_vector(cleaned):
        return ("vector", _parse_vector(cleaned))
    if _is_set(cleaned):
        return ("set", _parse_set_or_solutions(cleaned))
    if _is_solution_list(cleaned):
        return ("set", _parse_set_or_solutions(cleaned))
    if _is_interval(cleaned) or _is_interval_union(cleaned):
        return ("interval", _parse_interval(cleaned))
    if _has_top_level_or(cleaned) and not _has_relation(cleaned):
        return ("set", _parse_set_or_solutions(cleaned))
    if _has_relation(cleaned):
        return _parse_equation_or_relation(cleaned)
    if mode == "number":
        value = _parse_expr(cleaned)
        if not value.free_symbols and not value.is_number:
            raise ValueError(f"Couldn't read {text!r} as a number.")
        return ("expr", value)
    return ("expr", _parse_expr(cleaned))


def preprocess(raw: str) -> str:
    text = str(raw).strip()
    text = (
        text.replace("\u2212", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("×", "*")
        .replace("·", "*")
        .replace("∙", "*")
        .replace("≤", "<=")
        .replace("≥", ">=")
        .replace("≠", "!=")
        .replace("π", "pi")
        .replace("∞", "oo")
        .replace("√", "sqrt")
        .replace("∪", " U ")
        .replace("∩", " & ")
        .replace("⟨", "⟨")
        .replace("⟩", "⟩")
        .replace("∨", " or ")
        .replace("∅", "{}")
    )
    text = re.sub(r"^\s*\$\$(.*)\$\$\s*$", r"\1", text, flags=re.S)
    text = re.sub(r"^\s*\$(.*)\$\s*$", r"\1", text, flags=re.S)
    text = re.sub(r"^\s*\\\((.*)\\\)\s*$", r"\1", text, flags=re.S)
    text = re.sub(r"^\s*\\\[(.*)\\\]\s*$", r"\1", text, flags=re.S)
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\langle", "⟨").replace(r"\rangle", "⟩")
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    text = text.replace(r"\infty", "oo").replace(r"\infinity", "oo").replace(r"\inf", "oo")
    text = text.replace(r"\pi", "pi")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace(r"\leq", "<=").replace(r"\geq", ">=").replace(r"\le", "<=").replace(r"\ge", ">=")
    text = text.replace(r"\neq", "!=").replace(r"\ne", "!=")
    text = text.replace(r"\cup", " U ").replace(r"\cap", " & ")
    text = text.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    text = text.replace(r"\quad", " ").replace(r"\qquad", " ").replace(r"\ ", " ")
    text = re.sub(r"\\(?:mathrm|mathbf|mathit|operatorname|text)\{([^{}]*)\}", r"\1", text)
    for name in (
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "sin",
        "cos",
        "tan",
        "sec",
        "csc",
        "cot",
        "ln",
        "log",
        "exp",
        "sqrt",
    ):
        text = text.replace(f"\\{name}", name)
    text = _replace_sqrts(text)
    text = _replace_fracs(text)
    text = _replace_powers(text)
    text = text.replace("{", "{").replace("}", "}")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("⟨", "<").replace("⟩", ">")
    return text


def _replace_fracs(text: str) -> str:
    token = r"\\(?:d|t)?frac"
    while True:
        match = re.search(token, text)
        if not match:
            return text
        index = match.end()
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != "{":
            raise ValueError("A fraction is missing its numerator.")
        numerator, index = _take_brace(text, index)
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != "{":
            raise ValueError("A fraction is missing its denominator.")
        denominator, index = _take_brace(text, index)
        text = text[: match.start()] + f"(({numerator})/({denominator}))" + text[index:]


def _replace_sqrts(text: str) -> str:
    token = "sqrt"
    # Already-converted unicode radicals and latex \sqrt both land here as "sqrt".
    # Latex form is sqrt[n]{x} or sqrt{x}. Plain "sqrt(x)" is left for SymPy.
    pattern = re.compile(r"sqrt\s*(\[[^\]]+\])?\s*\{")
    while True:
        match = pattern.search(text)
        if not match:
            return text
        index = match.end() - 1
        body, end = _take_brace(text, index)
        root = match.group(1)
        if root:
            degree = root[1:-1].strip()
            replacement = f"(({body}))**(1/({degree}))"
        else:
            replacement = f"sqrt({body})"
        text = text[: match.start()] + replacement + text[end:]


def _replace_powers(text: str) -> str:
    while True:
        match = re.search(r"\^\{", text)
        if not match:
            return text
        body, end = _take_brace(text, match.end() - 1)
        text = text[: match.start()] + f"**({body})" + text[end:]


def _take_brace(text: str, index: int) -> tuple[str, int]:
    if index >= len(text) or text[index] != "{":
        raise ValueError("Expected a brace group in the math.")
    depth = 0
    for cursor in range(index, len(text)):
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
            if depth == 0:
                return text[index + 1 : cursor], cursor + 1
    raise ValueError("A brace group was left unclosed. Check the LaTeX.")


def _parse_expr(text: str) -> sp.Expr:
    prepared = _prepare_expr(text)
    try:
        value = parse_expr(prepared, transformations=_TRANSFORMS, evaluate=True)
    except Exception as exc:
        raise ValueError(_malformed(text, exc)) from exc
    if not isinstance(value, sp.Expr):
        raise ValueError(f"Couldn't read {text!r} as a mathematical expression.")
    return sp.sympify(value)


def _prepare_expr(text: str) -> str:
    prepared = text.strip()
    prepared = prepared.replace("^", "**")
    if prepared.endswith(("+", "-", "*", "/", "^", "=", "<", ">")):
        raise ValueError(f"Couldn't read {text!r}. It looks unfinished.")
    if prepared.count("(") != prepared.count(")"):
        raise ValueError(f"Couldn't read {text!r}. The parentheses don't match.")
    return prepared


def _malformed(text: str, exc: Exception) -> str:
    detail = str(exc).splitlines()[0].strip()
    if len(detail) > 160:
        detail = detail[:160] + "…"
    if detail:
        return f"Couldn't read {text!r} as math. {detail}"
    return f"Couldn't read {text!r} as math."


def _split_top(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and text.startswith(separator, index):
            parts.append(text[start:index])
            index += len(separator)
            start = index
            continue
        index += 1
    parts.append(text[start:])
    return parts


def _has_top_level(text: str, separator: str) -> bool:
    return len(_split_top(text, separator)) > 1


def _has_relation(text: str) -> bool:
    return any(_has_top_level(text, op) for op in ("=", ">=", "<=", ">", "<"))


def _has_top_level_or(text: str) -> bool:
    return bool(re.search(r"\bor\b", text, flags=re.I))


def _is_set(text: str) -> bool:
    return text.startswith("{") and text.endswith("}")


def _is_vector(text: str) -> bool:
    if not (text.startswith("<") and text.endswith(">") and "," in text):
        return False
    # "<" used as a comparison never wraps the whole answer with a closing ">".
    inner = text[1:-1]
    return "<" not in inner and ">" not in inner and all(part.strip() for part in _split_top(inner, ","))


def _is_interval(text: str) -> bool:
    if len(text) < 5 or text[0] not in "[(" or text[-1] not in "])":
        return False
    if text.startswith("<"):
        return False
    inner = text[1:-1]
    parts = _split_top(inner, ",")
    return len(parts) == 2 and all(part.strip() for part in parts)


def _is_interval_union(text: str) -> bool:
    pieces = _split_union(text)
    return len(pieces) > 1 and all(_is_interval(piece) for piece in pieces)


def _is_solution_list(text: str) -> bool:
    pieces = _split_or(text)
    if len(pieces) < 2:
        return False
    return all(_assignment_symbol(piece) for piece in pieces)


def _split_or(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s+\bor\b\s+", text, flags=re.I) if part.strip()]


def _split_union(text: str) -> list[str]:
    pieces = [part.strip() for part in _split_top(text, " U ") if part.strip()]
    return pieces


def _assignment_symbol(text: str) -> str | None:
    parts = _split_top(text, "=")
    if len(parts) != 2:
        return None
    left = parts[0].strip()
    if re.fullmatch(r"[A-Za-z]", left):
        return left
    return None


def _vector_inner(text: str) -> str | None:
    """Inner text of a point or vector, including parenthesized coordinates.

    An open interval uses the same parentheses, so this is only for answers
    already classified as points or vectors.
    """
    body = text.strip()
    if _is_vector(body):
        return body[1:-1]
    if body.startswith("(") and body.endswith(")") and "," in body:
        return body[1:-1]
    if body.startswith("[") and body.endswith("]") and "," in body and not _is_interval(body):
        return body[1:-1]
    return None


def _parse_vector(text: str) -> Tuple:
    inner = _vector_inner(text)
    if inner is None:
        raise ValueError(
            f"Couldn't read {text!r} as a point or vector. Use commas, like <1, -2, 3> or (1, -2)."
        )
    coords = [_parse_expr(part) for part in _split_top(inner, ",") if part.strip()]
    if len(coords) < 2:
        raise ValueError("A point or vector needs at least two components.")
    return Tuple(*coords)


def _parse_set_or_solutions(text: str) -> FiniteSet:
    body = text.strip()
    if _is_set(body):
        inner = body[1:-1].strip()
        if not inner:
            return FiniteSet()
        elements = [_parse_expr(part) for part in _split_top(inner, ",") if part.strip()]
        return FiniteSet(*elements)
    pieces = _split_or(body)
    if len(pieces) >= 2 and all(_assignment_symbol(piece) for piece in pieces):
        return FiniteSet(*(_parse_expr(_split_top(piece, "=")[1]) for piece in pieces))
    if "," in body and not _has_relation(body):
        return FiniteSet(*(_parse_expr(part) for part in _split_top(body, ",") if part.strip()))
    if len(pieces) >= 2:
        return FiniteSet(*(_parse_expr(piece) for piece in pieces))
    raise ValueError(f"Couldn't read {text!r} as a set.")


def _parse_endpoint(text: str) -> sp.Expr:
    token = text.strip().lower()
    if token in {"oo", "infinity", "inf", "+oo", "+infinity"}:
        return sp.oo
    if token in {"-oo", "-infinity", "-inf"}:
        return -sp.oo
    return _parse_expr(text)


def _parse_one_interval(text: str) -> Interval:
    if not _is_interval(text):
        raise ValueError(
            f"Couldn't read {text!r} as an interval. Use brackets, like [0, 1) or (-oo, 2]."
        )
    left_open = text[0] == "("
    right_open = text[-1] == ")"
    left, right = _split_top(text[1:-1], ",")
    lo = _parse_endpoint(left)
    hi = _parse_endpoint(right)
    try:
        return Interval(lo, hi, left_open, right_open)
    except Exception as exc:
        raise ValueError(f"Couldn't read {text!r} as an interval. {exc}") from exc


def _parse_interval(text: str) -> Interval | Union:
    pieces = _split_union(text)
    if len(pieces) == 1:
        return _parse_one_interval(pieces[0])
    return Union(*(_parse_one_interval(piece) for piece in pieces))


def _parse_equation_or_relation(text: str) -> tuple[str, Any]:
    compound = _parse_compound_inequality(text)
    if compound is not None:
        return ("interval", compound)
    pieces = _split_or(text)
    if len(pieces) >= 2 and all(_assignment_symbol(piece) for piece in pieces):
        return ("set", _parse_set_or_solutions(text))
    for op in ("=", ">=", "<=", ">", "<"):
        parts = _split_top(text, op)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            left = _parse_expr(parts[0])
            right = _parse_expr(parts[1])
            if op == "=":
                return ("equation", sp.Eq(left, right, evaluate=False))
            relation = {">=": sp.Ge, "<=": sp.Le, ">": sp.Gt, "<": sp.Lt}[op]
            return ("relation", relation(left, right, evaluate=False))
    raise ValueError(f"Couldn't read {text!r} as an equation or inequality.")


def _parse_compound_inequality(text: str) -> Interval | None:
    match = re.fullmatch(
        r"\s*(.+?)\s*(<=|<|>=|>)\s*(.+?)\s*(<=|<|>=|>)\s*(.+)\s*",
        text,
    )
    if not match:
        return None
    left, op1, middle, op2, right = match.groups()
    if not re.fullmatch(r"[A-Za-z]", middle.strip()):
        return None
    symbol = sp.Symbol(middle.strip())
    lower_op, upper_op = op1, op2
    lower, upper = _parse_expr(left), _parse_expr(right)
    # Only increasing chains: a < x < b or a > x > b (which flips).
    ascending = {">", ">="}
    descending = {"<", "<="}
    if op1 in descending and op2 in descending:
        lo, hi = lower, upper
        left_open = op1 == "<"
        right_open = op2 == "<"
    elif op1 in ascending and op2 in ascending:
        lo, hi = upper, lower
        left_open = op2 == ">"
        right_open = op1 == ">"
    else:
        return None
    return Interval(lo, hi, left_open, right_open)


def _kind_for_pair(expected: str, given: str, kind: str) -> str:
    if kind != "auto":
        return kind
    exp = preprocess(expected)
    got = preprocess(given)
    if _is_vector(exp) and got.startswith("(") and got.endswith(")") and "," in got:
        return "vector"
    if _is_vector(got) and exp.startswith("(") and exp.endswith(")") and "," in exp:
        return "vector"
    return "auto"


def _align_solutions(
    expected: tuple[str, Any], given: tuple[str, Any]
) -> tuple[tuple[str, Any], tuple[str, Any]]:
    if expected[0] == "set" and given[0] == "set":
        return expected, given
    if expected[0] == "set" and given[0] == "equation":
        values = _equation_solution_values(given[1])
        if values is not None:
            return expected, ("set", FiniteSet(*values))
    if given[0] == "set" and expected[0] == "equation":
        values = _equation_solution_values(expected[1])
        if values is not None:
            return ("set", FiniteSet(*values)), given
    return expected, given


def _equation_solution_values(equation: sp.Eq) -> list[sp.Expr] | None:
    if isinstance(equation.lhs, sp.Symbol) and equation.lhs not in equation.rhs.free_symbols:
        return [equation.rhs]
    if isinstance(equation.rhs, sp.Symbol) and equation.rhs not in equation.lhs.free_symbols:
        return [equation.lhs]
    return None


def _coerce_pair(
    expected: tuple[str, Any], given: tuple[str, Any]
) -> tuple[tuple[str, Any], tuple[str, Any]] | None:
    pairs = (expected, given)
    kinds = {expected[0], given[0]}
    if kinds == {"relation", "interval"}:
        relation = expected if expected[0] == "relation" else given
        interval = expected if expected[0] == "interval" else given
        converted = _relation_to_interval(relation[1])
        if converted is None:
            return None
        if expected[0] == "relation":
            return ("interval", converted), interval
        return interval, ("interval", converted)
    if kinds == {"equation", "expr"}:
        equation = expected[1] if expected[0] == "equation" else given[1]
        expr = expected[1] if expected[0] == "expr" else given[1]
        values = _equation_solution_values(equation)
        if values is not None and len(values) == 1:
            if expected[0] == "equation":
                return ("expr", values[0]), ("expr", expr)
            return ("expr", expr), ("expr", values[0])
    if kinds == {"vector", "expr"}:
        return None
    _ = pairs
    return None


def _relation_to_interval(relation: sp.Relational) -> Interval | None:
    symbol = None
    for candidate in (relation.lhs, relation.rhs):
        if isinstance(candidate, sp.Symbol):
            symbol = candidate
            break
    if symbol is None or len(relation.free_symbols) != 1:
        return None
    solved = sp.solve(relation, symbol)
    if isinstance(solved, sp.Set):
        if isinstance(solved, (Interval, Union)):
            return solved
    return None


def _equivalent(
    expected: tuple[str, Any],
    given: tuple[str, Any],
    tolerance: float,
    places: int | None,
    expected_raw: str,
    given_raw: str,
    notes: list[str],
) -> bool:
    kind, left, right = expected[0], expected[1], given[1]
    if kind == "expr":
        return _status_ok(_values_match(left, right, given_raw, places, tolerance), places, notes)
    if kind == "vector":
        got_parts = _component_texts(given_raw)
        if len(left) != len(right) or got_parts is None or len(got_parts) != len(right):
            return False
        statuses = [
            _values_match(exp, got, raw, places, tolerance)
            for exp, got, raw in zip(left, right, got_parts)
        ]
        return _combine_statuses(statuses, places, notes)
    if kind == "set":
        if _sets_equivalent(left, right, tolerance):
            return True
        return _members_decimal(expected_raw, given_raw, places, tolerance, notes)
    if kind == "interval":
        if _sets_equivalent(left, right, tolerance):
            return True
        return _intervals_decimal(expected_raw, given_raw, places, tolerance, notes)
    if kind == "equation":
        if _equations_equivalent(left, right, tolerance):
            return True
        return _equation_decimal(left, right, given_raw, places, tolerance, notes)
    if kind == "relation":
        return _relations_equivalent(left, right)
    raise ValueError(f"Cannot compare {kind} answers.")


def _status_ok(status: str, places: int | None, notes: list[str]) -> bool:
    if status == "ok":
        return True
    if status == "too_few":
        notes.append(_too_few_message(places))
    return False


def _combine_statuses(statuses: list[str], places: int | None, notes: list[str]) -> bool:
    if statuses and all(status == "ok" for status in statuses):
        return True
    if any(status == "no" for status in statuses):
        return False
    if any(status == "too_few" for status in statuses):
        notes.append(_too_few_message(places))
    return False


def _too_few_message(places: int | None) -> str:
    count = 3 if places is None else places
    word = "place" if count == 1 else "places"
    return (
        f"Not quite. That decimal needs {count} {word} after the decimal point. "
        "An exact value is fine too."
    )


def _values_match(
    expected: sp.Expr,
    given: sp.Expr,
    given_raw: str,
    places: int | None,
    tolerance: float,
) -> str:
    if _exprs_equivalent(expected, given, tolerance):
        return "ok"
    if places is None or not _is_numeric(expected) or not _is_numeric(given):
        return "no"
    verdict = _grade_literal(expected, given_raw, places)
    if verdict == "ok":
        return "ok"
    if verdict == "too_few":
        return "too_few"
    return "no"


_DECIMAL_LITERAL = re.compile(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)")


def _decimal_literal(text: str) -> tuple[Decimal, int] | None:
    token = text.strip().replace(" ", "")
    if not re.fullmatch(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)", token):
        return None
    if "." in token.lstrip("+-"):
        places = len(token.lstrip("+-").split(".", 1)[1])
    else:
        places = 0
    return Decimal(token), places


def _exact_decimal(expr: sp.Expr) -> tuple[Decimal, int | None] | None:
    """Return the exact value and how many places it takes to terminate.

    ``None`` for the place count means the value does not terminate.
    """
    simplified = sp.simplify(expr)
    if not _is_numeric(simplified):
        return None
    if simplified in (sp.oo, -sp.oo) or simplified.has(sp.oo):
        return None
    if bool(getattr(simplified, "is_rational", False)):
        rational = sp.Rational(simplified)
        with localcontext() as ctx:
            ctx.prec = 80
            value = Decimal(int(rational.p)) / Decimal(int(rational.q))
        return value, _terminating_places(rational)
    shown = str(sp.N(simplified, 70))
    if "." in shown and "e" not in shown.lower():
        shown = shown[:-1]
    return Decimal(shown), None


def _terminating_places(rational: sp.Rational) -> int | None:
    denominator = abs(int(rational.q))
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return None
    return max(twos, fives)


def _chop(value: Decimal, places: int, rounding: str) -> Decimal:
    quant = Decimal(1).scaleb(-places) if places else Decimal(1)
    with localcontext() as ctx:
        ctx.prec = max(80, places + 20)
        ctx.rounding = rounding
        return +value.quantize(quant)


def _grade_literal(expected: sp.Expr, given_raw: str, places: int) -> str:
    parsed = _decimal_literal(given_raw)
    exact = _exact_decimal(expected)
    if parsed is None or exact is None:
        return "not_decimal"
    student_value, written = parsed
    exact_value, terminated = exact
    if student_value == exact_value:
        return "ok"
    if written < places:
        if terminated is not None and written >= terminated and student_value == exact_value:
            return "ok"
        rounded = _chop(exact_value, written, ROUND_HALF_UP)
        truncated = _chop(exact_value, written, ROUND_DOWN)
        if student_value == rounded or student_value == truncated:
            return "too_few"
        return "mismatch"
    rounded = _chop(exact_value, written, ROUND_HALF_UP)
    truncated = _chop(exact_value, written, ROUND_DOWN)
    if student_value == rounded or student_value == truncated:
        return "ok"
    return "mismatch"


def _component_texts(text: str) -> list[str] | None:
    inner = _vector_inner(text)
    if inner is None:
        return None
    parts = [part.strip() for part in _split_top(inner, ",") if part.strip()]
    return parts or None


def _member_texts(text: str) -> list[str] | None:
    body = text.strip()
    if _is_set(body):
        inner = body[1:-1].strip()
        if not inner:
            return []
        return [part.strip() for part in _split_top(inner, ",") if part.strip()]
    pieces = _split_or(body)
    if len(pieces) >= 2 and all(_assignment_symbol(piece) for piece in pieces):
        return [_split_top(piece, "=")[1].strip() for piece in pieces]
    return None


def _members_decimal(
    expected_raw: str,
    given_raw: str,
    places: int | None,
    tolerance: float,
    notes: list[str],
) -> bool:
    expected_parts = _member_texts(expected_raw)
    given_parts = _member_texts(given_raw)
    if expected_parts is None or given_parts is None or len(expected_parts) != len(given_parts):
        return False
    remaining = given_parts[:]
    statuses: list[str] = []
    for part in expected_parts:
        expected_expr = _parse_expr(part)
        found = None
        found_status = "no"
        for candidate in remaining:
            status = _values_match(expected_expr, _parse_expr(candidate), candidate, places, tolerance)
            if status in {"ok", "too_few"}:
                found = candidate
                found_status = status
                break
        if found is None:
            return False
        remaining.remove(found)
        statuses.append(found_status)
    return _combine_statuses(statuses, places, notes)


def _interval_bounds(text: str) -> list[tuple[str, str, bool, bool]] | None:
    pieces = _split_union(text.strip())
    if not pieces or not all(_is_interval(piece) for piece in pieces):
        return None
    bounds = []
    for piece in pieces:
        left, right = _split_top(piece[1:-1], ",")
        bounds.append((left.strip(), right.strip(), piece[0] == "(", piece[-1] == ")"))
    return bounds


def _endpoint_status(expected_raw: str, given_raw: str, places: int | None, tolerance: float) -> str:
    expected_inf = _infinity_sign(expected_raw)
    given_inf = _infinity_sign(given_raw)
    if expected_inf is not None or given_inf is not None:
        return "ok" if expected_inf == given_inf else "no"
    return _values_match(_parse_endpoint(expected_raw), _parse_endpoint(given_raw), given_raw, places, tolerance)


def _infinity_sign(text: str) -> int | None:
    token = text.strip().lower()
    if token in {"oo", "infinity", "inf", "+oo", "+infinity"}:
        return 1
    if token in {"-oo", "-infinity", "-inf"}:
        return -1
    return None


def _intervals_decimal(
    expected_raw: str,
    given_raw: str,
    places: int | None,
    tolerance: float,
    notes: list[str],
) -> bool:
    expected_bounds = _interval_bounds(expected_raw)
    given_bounds = _interval_bounds(given_raw)
    if expected_bounds is None or given_bounds is None or len(expected_bounds) != len(given_bounds):
        return False
    remaining = given_bounds[:]
    statuses: list[str] = []
    for left, right, left_open, right_open in expected_bounds:
        found = None
        found_status = "no"
        for candidate in remaining:
            cleft, cright, cleft_open, cright_open = candidate
            if cleft_open != left_open or cright_open != right_open:
                continue
            pair = _combine_pair(
                _endpoint_status(left, cleft, places, tolerance),
                _endpoint_status(right, cright, places, tolerance),
            )
            if pair in {"ok", "too_few"}:
                found = candidate
                found_status = pair
                break
        if found is None:
            return False
        remaining.remove(found)
        statuses.append(found_status)
    return _combine_statuses(statuses, places, notes)


def _combine_pair(left: str, right: str) -> str:
    if left == "no" or right == "no":
        return "no"
    if left == "too_few" or right == "too_few":
        return "too_few"
    return "ok"


def _equation_decimal(
    expected: sp.Eq,
    given: sp.Eq,
    given_raw: str,
    places: int | None,
    tolerance: float,
    notes: list[str],
) -> bool:
    expected_value = _lone_number(expected)
    given_value = _lone_number(given)
    if expected_value is None or given_value is None or "=" not in given_raw:
        return False
    given_side = _split_top(given_raw, "=")[-1]
    return _status_ok(_values_match(expected_value, given_value, given_side, places, tolerance), places, notes)


def _lone_number(equation: sp.Eq) -> sp.Expr | None:
    values = _equation_solution_values(equation)
    if values is None or len(values) != 1 or not _is_numeric(values[0]):
        return None
    return values[0]


def _format_places(value: Decimal, places: int, rounding: str) -> str:
    chopped = _chop(value, places, rounding)
    return f"{chopped:.{places}f}"


def _preview_number(expr: sp.Expr, places: int) -> str | None:
    exact = _exact_decimal(expr)
    if exact is None:
        return None
    value, _terminated = exact
    rounded = _format_places(value, places, ROUND_HALF_UP)
    truncated = _format_places(value, places, ROUND_DOWN)
    if rounded == truncated:
        return rounded
    return f"{rounded} or {truncated}"


def _preview_value(text: str, tagged: tuple[str, Any], places: int) -> str | None:
    kind, value = tagged
    if kind == "expr" and _is_numeric(value):
        return _preview_number(value, places)
    if kind == "vector":
        shown = [_preview_number(item, places) for item in value]
        if any(item is None for item in shown):
            return None
        opener, closer = ("<", ">") if text.strip().startswith("<") else ("(", ")")
        return opener + ", ".join(shown) + closer
    if kind == "set" and isinstance(value, FiniteSet):
        parts = _member_texts(text)
        if parts is None:
            return None
        shown = []
        for part in parts:
            try:
                expr = _parse_expr(part)
            except ValueError:
                return None
            preview = _preview_number(expr, places)
            if preview is None:
                return None
            shown.append(preview)
        return "{" + ", ".join(shown) + "}"
    if kind == "interval":
        bounds = _interval_bounds(text)
        if bounds is None:
            return None
        pieces = []
        for left, right, left_open, right_open in bounds:
            left_shown = _preview_endpoint(left, places)
            right_shown = _preview_endpoint(right, places)
            if left_shown is None or right_shown is None:
                return None
            pieces.append(
                f"{'(' if left_open else '['}{left_shown}, {right_shown}{')' if right_open else ']'}"
            )
        return " U ".join(pieces)
    if kind == "equation":
        number = _lone_number(value)
        if number is None:
            return None
        return _preview_number(number, places)
    return None


def _preview_endpoint(text: str, places: int) -> str | None:
    if _infinity_sign(text) is not None:
        return text.strip()
    try:
        return _preview_number(_parse_endpoint(text), places)
    except ValueError:
        return None


def _numbers_close(left: sp.Expr, right: sp.Expr, tolerance: float) -> bool:
    try:
        difference = abs(complex(sp.N(left - right)))
        scale = max(1.0, abs(complex(sp.N(left))), abs(complex(sp.N(right))))
    except Exception:
        return False
    if abs(difference.imag) > tolerance:
        return False
    return abs(difference.real) <= tolerance or abs(difference.real) <= tolerance * scale


def _exprs_equivalent(left: sp.Expr, right: sp.Expr, tolerance: float) -> bool:
    left = sp.sympify(left)
    right = sp.sympify(right)
    for candidate in (
        left - right,
        sp.expand(left - right),
        sp.simplify(left - right),
        sp.radsimp(left) - sp.radsimp(right),
        sp.sqrtdenest(left) - sp.sqrtdenest(right),
        sp.trigsimp(left - right),
        sp.expand(sp.radsimp(left)) - sp.expand(sp.radsimp(right)),
    ):
        try:
            if sp.simplify(candidate) == 0:
                return True
        except Exception:
            continue
    if getattr(left, "free_symbols", set()) or getattr(right, "free_symbols", set()):
        return _sample_equivalent(left, right, tolerance)
    return False


def _sample_equivalent(left: sp.Expr, right: sp.Expr, tolerance: float) -> bool:
    symbols = sorted(left.free_symbols | right.free_symbols, key=str)
    if not symbols:
        return False
    samples = (-3, -1, -0.5, 0.5, 1, 2, 4)
    hits = 0
    for index in range(8):
        values = {
            symbol: samples[(index + offset) % len(samples)]
            for offset, symbol in enumerate(symbols)
        }
        try:
            got_left = complex(left.subs(values).doit().evalf())
            got_right = complex(right.subs(values).doit().evalf())
        except Exception:
            continue
        if got_left.imag or got_right.imag:
            if abs(got_left - got_right) > 1e-6:
                return False
        elif abs(got_left - got_right) > max(tolerance, 1e-6) * max(1.0, abs(got_left), abs(got_right)):
            return False
        hits += 1
    return hits >= 4


def _sets_equivalent(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, FiniteSet) and isinstance(right, FiniteSet):
        if len(left) != len(right):
            return False
        remaining = list(right)
        for item in left:
            found = None
            for candidate in remaining:
                if _exprs_equivalent(item, candidate, tolerance):
                    found = candidate
                    break
            if found is None:
                return False
            remaining.remove(found)
        return True
    try:
        if sp.simplify(left.symmetric_difference(right)) == sp.EmptySet or left == right:
            return True
    except Exception:
        pass
    return left == right


def _equations_equivalent(left: sp.Eq, right: sp.Eq, tolerance: float) -> bool:
    first = sp.simplify(sp.expand(left.lhs - left.rhs))
    second = sp.simplify(sp.expand(right.lhs - right.rhs))
    if first == 0 and second == 0:
        return True
    if first == 0 or second == 0:
        return False
    try:
        ratio = sp.simplify(sp.together(first / second))
        if ratio.free_symbols == set() and ratio != 0:
            return True
    except Exception:
        pass
    if _sample_equivalent(first, second, tolerance):
        # Same zero set can still differ by a non-constant factor; require a constant ratio numerically.
        symbols = sorted(first.free_symbols | second.free_symbols, key=str)
        ratios = []
        for index, sample in enumerate((-2, -0.5, 1, 3)):
            values = {symbol: sample + index for symbol in symbols}
            try:
                numerator = complex(first.subs(values).evalf())
                denominator = complex(second.subs(values).evalf())
            except Exception:
                return False
            if abs(denominator) < 1e-8:
                continue
            ratios.append(numerator / denominator)
        if len(ratios) >= 2 and all(abs(ratio - ratios[0]) < 1e-4 for ratio in ratios[1:]):
            return abs(ratios[0]) > 1e-8
    return False


def _relations_equivalent(left: sp.Relational, right: sp.Relational) -> bool:
    symbols = left.free_symbols | right.free_symbols
    if len(symbols) != 1:
        return left == right
    symbol = next(iter(symbols))
    return _same_solution(left, right, symbol)


def _same_solution(left: sp.Relational, right: sp.Relational, symbol: sp.Symbol) -> bool:
    try:
        return sp.solve(left, symbol) == sp.solve(right, symbol)
    except Exception:
        return False


def _categorize(expected: tuple[str, Any], given: tuple[str, Any], tolerance: float) -> str:
    kind = expected[0]
    left, right = expected[1], given[1]
    if kind == "vector" and len(left) != len(right):
        return "dimension_mismatch"
    if kind == "vector":
        if all(_exprs_equivalent(-a, b, tolerance) for a, b in zip(left, right)):
            return "sign_error"
        return "algebra_slip"
    if kind == "set" and isinstance(left, FiniteSet) and isinstance(right, FiniteSet):
        if len(right) < len(left) and _is_subset_tolerant(right, left, tolerance):
            return "incomplete"
        if len(right) > len(left) and _is_subset_tolerant(left, right, tolerance):
            return "extraneous"
        return "algebra_slip"
    if kind == "expr":
        if _exprs_equivalent(-left, right, tolerance) or (
            _is_numeric(left) and _is_numeric(right) and _numbers_close(-left, right, tolerance)
        ):
            return "sign_error"
        if _is_numeric(left) and _is_numeric(right):
            return "arithmetic"
        try:
            ratio = sp.simplify(right / left)
            if ratio.free_symbols == set() and ratio not in (0, 1, sp.zoo, sp.nan):
                return "off_by_factor"
        except Exception:
            pass
        try:
            difference = sp.simplify(sp.expand(right - left))
            if difference.free_symbols == set() and difference != 0:
                return "off_by_constant"
        except Exception:
            pass
        return "algebra_slip"
    if kind == "equation":
        first = sp.expand(left.lhs - left.rhs)
        second = sp.expand(right.lhs - right.rhs)
        if _exprs_equivalent(first, -second, tolerance):
            return "sign_error"
        return "algebra_slip"
    if kind in {"interval", "relation"}:
        return "interval_error"
    return "algebra_slip"


def _is_subset_tolerant(small: FiniteSet, large: FiniteSet, tolerance: float) -> bool:
    for item in small:
        if not any(_exprs_equivalent(item, other, tolerance) for other in large):
            return False
    return True


def _is_numeric(value: sp.Expr) -> bool:
    return not getattr(value, "free_symbols", set()) and bool(getattr(value, "is_number", False))


def _student_message(category: str) -> str:
    messages = {
        "sign_error": "Not quite. Check the sign.",
        "off_by_factor": "Not quite. The form is close, but a factor is off.",
        "off_by_constant": "Not quite. A constant term doesn't match.",
        "incomplete": "Not quite. That looks like only part of the answer.",
        "extraneous": "Not quite. That includes something that doesn't belong.",
        "dimension_mismatch": "Not quite. The number of components doesn't match.",
        "arithmetic": "Not quite. Check the arithmetic.",
        "interval_error": "Not quite. Check the endpoints and whether they are included.",
        "form_mismatch": "That doesn't match the kind of answer this problem expects.",
        "algebra_slip": "Not equivalent.",
    }
    return messages.get(category, "Not equivalent.")
