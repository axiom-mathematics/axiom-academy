"""Exact mathematical answer checking.

SymPy decides correctness. An LLM is never consulted. The same module is
copied to ``student-pages/mathcheck.py`` and executed in the browser with
Pyodide, so keep this file free of application imports.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def check_answer(
    expected: str,
    given: str,
    *,
    tolerance: float = 1e-3,
    kind: str = "auto",
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

    try:
        same = _equivalent(exp_value, got_value, tolerance)
    except ValueError as exc:
        return CheckResult("invalid", str(exc), equivalent=None)

    if same:
        return CheckResult("correct", "That's equivalent.", error_category=None, equivalent=True)

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


def _parse_vector(text: str) -> Tuple:
    body = text.strip()
    if _is_vector(body):
        inner = body[1:-1]
    elif body.startswith("(") and body.endswith(")") and "," in body:
        inner = body[1:-1]
    elif body.startswith("[") and body.endswith("]") and "," in body and not _is_interval(body):
        inner = body[1:-1]
    else:
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


def _equivalent(expected: tuple[str, Any], given: tuple[str, Any], tolerance: float) -> bool:
    kind, left, right = expected[0], expected[1], given[1]
    if kind == "expr":
        return _exprs_equivalent(left, right, tolerance)
    if kind == "vector":
        if len(left) != len(right):
            return False
        return all(_exprs_equivalent(a, b, tolerance) for a, b in zip(left, right))
    if kind == "set":
        return _sets_equivalent(left, right, tolerance)
    if kind == "interval":
        return _sets_equivalent(left, right, tolerance)
    if kind == "equation":
        return _equations_equivalent(left, right, tolerance)
    if kind == "relation":
        return _relations_equivalent(left, right)
    raise ValueError(f"Cannot compare {kind} answers.")


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
    return _numbers_close(left, right, tolerance)


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
