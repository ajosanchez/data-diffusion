# AGENTS.md

## Python dependencies

- Never use `pip` or `uv pip install`. Add/remove packages only with `uv add <pkg>` / `uv remove <pkg>`.
- Run Python with `uv run python ...`. Never activate the venv (`source .venv/bin/activate`) — uv manages it automatically.
- Prefer the Python standard library. Only reach for a third-party package if it saves substantial effort.
- Any third-party dependency requires the user's explicit approval first. Before asking, research the package: GitHub stars, maintenance activity (recent commits/releases), and whether it's outdated or superseded by a better alternative.

## Python Architecture & Abstraction Guidelines

- **Default to Plain Functions and Built-ins:** Prefer top-level functions, dictionaries, tuples, lists, and standard library tools (`dataclasses`, `NamedTuple`) over custom class hierarchies.
- **The "Two Method" Rule (Jack Diederich):** If a class has only `__init__` and one other method, replace it with a simple function. Do not create single-method classes or "Manager/Helper/Handler" wrappers.
- **Justify Every Class:** Create a class only when:
  - There is genuine, non-trivial mutable state to encapsulate.
  - It dramatically simplifies readability, public API boundaries, or unit testing/mocking.
  - Polymorphism/subtyping is strictly necessary (prefer structural typing/`typing.Protocol` when needed).
- **Rule of Three:** Do not introduce an abstraction (classes, interfaces, generic factories) for hypothetical future extensibility. Write concrete, straightforward code first.

## Code Style & Type Annotations (Python 3.12+)

- **Strict Modern Type Hints:**
  - Fully type all function signatures, including arguments and return types (`-> ReturnType` or `-> None`).
  - Use Python 3.12+ syntax: built-in generics (`list[str]`, `dict[str, int]`), union operators (`X | None`), and modern type aliases (`type Point = tuple[float, float]`).
  - Add explicit type annotations to local variables whenever the inferred type is ambiguous or non-obvious (e.g., `records: list[dict[str, Any]] = fetch_data()`).

- **Concise Docstrings:**
  - Add docstrings to public interfaces, complex logic, and developer-facing APIs. Omit them for trivial, self-explanatory one-liners.
  - Keep docstrings extremely compact to conserve tokens. Avoid verbose boilerplate.
  - Format: A 1-sentence summary, followed directly by concise `Args:` and `Returns:` sections. Types belong in the signature—do not repeat type names in docstring descriptions unless clarifying structure.

**Docstring Example:**
```python
def process_batch(items: list[str], max_retries: int = 3) -> dict[str, bool]:
    """Processes batch items sequentially and reports success status.

    Args:
        items: Raw item identifiers to dispatch.
        max_retries: Number of retry attempts per failed item.

    Returns:
        Mapping of item identifier to completion boolean.
    """
```

## Testing Strategy (`pytest`)

- **Framework:** Use `pytest` for all unit and integration testing.
- **TDD / Test-First Approach:**
  - Write test cases **before** writing the implementation.
  - Verify the test fails for the expected reason (Red) before writing code to make it pass (Green).
- **Substantive Behavior over Trivial Checks:**
  - Never write "smoke-only" or tautological tests (e.g., verifying an import succeeds, checking that a mock was called once without inspecting state, or testing trivial variable assignments).
  - Test the **contract and invariants**: verify that inputs produce exact expected outputs, state changes persist correctly, and business logic behaves properly under nominal and failure conditions.
- **Edge Cases & Error Paths:**
  - Every test suite must cover boundary values, empty states, invalid inputs, and explicit exceptions (using `pytest.raises(...)`).
- **Isolation & Parameterization:**
  - Use `pytest.mark.parametrize` to test multiple input/output scenarios cleanly without boilerplate.
  - Rely on lightweight standard pytest fixtures (`tmp_path`, `monkeypatch`) instead of complex, brittle mock hierarchies.

## Checks

When done writing code, run all of the following and fix any issues they report:

1. Type checking: `uv run ty check`
2. Formatting: `uv run ruff format .`
3. Linting: `uv run ruff check .`
4. Tests: `uv run pytest`
