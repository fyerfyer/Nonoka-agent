"""
Prompt Template System

A lightweight but powerful prompt templating engine built on Jinja2.
Supports function-style templates, file-based templates, and template
composition — all with full type safety.

Usage — Function-style (simplest)::

    from nonoka import prompt

    @prompt
    async def code_review(diff: str, language: str = "python") -> str:
        return f"Review this {language} code:\n```\n{diff}\n```"

    system_prompt = await code_review(diff="...", language="go")

Usage — Jinja2 Template (most flexible)::

    from nonoka.prompt import PromptTemplate

    tmpl = PromptTemplate.from_string('''
    You are a {{role}}. Review this {{language}} code:
    ```{{language}}
    {{code}}
    ```
    {% if strict %}
    Be extremely strict about style violations.
    {% endif %}
    ''')

    output = await tmpl.render(role="senior engineer", language="python", code="...", strict=True)

Usage — File-based::

    tmpl = PromptTemplate.from_file("prompts/code_review.j2")
    output = await tmpl.render(...)

Usage — Template composition::

    base = PromptTemplate.from_string("You are {{role}}.\n\n{{content}}")
    review = PromptTemplate.from_string("Review: {{code}}")

    output = await base.render(role="engineer", content=await review.render(code="..."))
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
  Any,
  Awaitable,
  Callable,
  Generic,
  ParamSpec,
  TypeVar,
  overload,
)

import jinja2

# --------------------------------------------------------------------------- #
# Jinja2 environment (shared, with safe defaults)
# --------------------------------------------------------------------------- #

_default_jinja_env = jinja2.Environment(
  loader=jinja2.BaseLoader(),
  autoescape=False,  # Prompts are not HTML
  trim_blocks=True,
  lstrip_blocks=True,
  keep_trailing_newline=True,
)


# --------------------------------------------------------------------------- #
# PromptTemplate — Jinja2-based template
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PromptTemplate:
  """Immutable Jinja2 prompt template.

  Supports both synchronous and asynchronous rendering.  Templates are
  parsed once at construction time and can be reused safely across
  coroutines (immutable).
  """

  source: str
  _template: jinja2.Template = field(repr=False, compare=False)

  def __init__(self, source: str):
    object.__setattr__(self, "source", source)
    object.__setattr__(self, "_template", _default_jinja_env.from_string(source))

  # ------------------------------------------------------------------ #
  # Factory methods
  # ------------------------------------------------------------------ #

  @classmethod
  def from_string(cls, source: str) -> PromptTemplate:
    """Create a template from a raw string."""
    return cls(source)

  @classmethod
  def from_file(cls, path: str | Path) -> PromptTemplate:
    """Load a template from a file path."""
    path = Path(path)
    if not path.exists():
      raise FileNotFoundError(f"Prompt template file not found: {path}")
    return cls(path.read_text(encoding="utf-8"))

  # ------------------------------------------------------------------ #
  # Rendering
  # ------------------------------------------------------------------ #

  def render_sync(self, **kwargs: Any) -> str:
    """Synchronous render — safe to call from sync code."""
    return self._template.render(**kwargs)

  async def render(self, **kwargs: Any) -> str:
    """Asynchronous render (calls render_sync in thread pool)."""
    return self.render_sync(**kwargs)

  def __call__(self, **kwargs: Any) -> str:
    """Shorthand for ``render_sync``."""
    return self.render_sync(**kwargs)

  def partial(self, **preset: Any) -> "PartialPromptTemplate":
    """Create a partially-bound template (currying).

    Useful for defining a template skeleton and filling in details later::

        base = PromptTemplate("You are a {{role}}.\n\n{{content}}").partial(role="coder")
        out = base.render(content="Write a function.")
    """
    return PartialPromptTemplate(self, preset)


# --------------------------------------------------------------------------- #
# PartialPromptTemplate — curried / partially-bound template
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PartialPromptTemplate:
  """A template with some variables already bound."""

  _template: PromptTemplate
  _preset: dict[str, Any] = field(default_factory=dict)

  def render_sync(self, **kwargs: Any) -> str:
    merged = {**self._preset, **kwargs}
    return self._template.render_sync(**merged)

  async def render(self, **kwargs: Any) -> str:
    merged = {**self._preset, **kwargs}
    return await self._template.render(**merged)

  def __call__(self, **kwargs: Any) -> str:
    return self.render_sync(**kwargs)

  def partial(self, **preset: Any) -> "PartialPromptTemplate":
    """Further partial binding."""
    merged = {**self._preset, **preset}
    return PartialPromptTemplate(self._template, merged)


# --------------------------------------------------------------------------- #
# @prompt decorator — function-style templates
# --------------------------------------------------------------------------- #

P = ParamSpec("P")
T = TypeVar("T")


class PromptFunction(Generic[P]):
  """A function wrapped as a prompt template.

  If the function body returns a string, that value is used as the prompt.
  Otherwise, the function's docstring is treated as a Jinja2 template and
  rendered with the bound arguments.
  """

  def __init__(self, fn: Callable[P, str | Awaitable[str]]):
    self._fn = fn
    self._is_async = inspect.iscoroutinefunction(fn)
    self.__doc__ = fn.__doc__
    self.__name__ = getattr(fn, "__name__", "prompt")

    # Store docstring for use as a format template
    self._doc = fn.__doc__ or ""

  @property
  def fn(self) -> Callable[P, str | Awaitable[str]]:
    return self._fn

  def _bind_args(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
    """Bind positional + keyword args to parameter names."""
    sig = inspect.signature(self._fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)

  def _render_template(self, *args: P.args, **kwargs: P.kwargs) -> str:
    """Render docstring template with bound arguments using str.format()."""
    if not self._doc:
      return ""
    ctx = self._bind_args(*args, **kwargs)
    try:
      return self._doc.format(**ctx)
    except (KeyError, ValueError):
      # If format fails (e.g. user wants literal braces), return raw docstring
      return self._doc

  def render_sync(self, *args: P.args, **kwargs: P.kwargs) -> str:
    """Render synchronously (only works if the underlying function is sync)."""
    if self._is_async:
      raise RuntimeError(
        f"Prompt function '{self.__name__}' is async. "
        "Use 'await ...render(...)' or 'asyncio.run()' instead."
      )
    result = self._fn(*args, **kwargs)
    if isinstance(result, str):
      return result
    # Fallback: render docstring template
    return self._render_template(*args, **kwargs)

  async def render(self, *args: P.args, **kwargs: P.kwargs) -> str:
    """Render asynchronously (works for both sync and async functions)."""
    if self._is_async:
      result = await self._fn(*args, **kwargs)  # type: ignore[misc]
      if isinstance(result, str):
        return result
      return self._render_template(*args, **kwargs)
    result = self._fn(*args, **kwargs)
    if isinstance(result, str):
      return result
    return self._render_template(*args, **kwargs)

  def __call__(self, *args: P.args, **kwargs: P.kwargs) -> str:
    if self._is_async:
      raise RuntimeError(
        f"Prompt function '{self.__name__}' is async. "
        "Use 'await ...render(...)' or 'asyncio.run()' instead."
      )
    return self.render_sync(*args, **kwargs)


@overload
def prompt(fn: Callable[P, str | Awaitable[str]]) -> PromptFunction[P]: ...


@overload
def prompt(
  *,
  description: str | None = None,
) -> Callable[[Callable[P, str | Awaitable[str]]], PromptFunction[P]]: ...


def prompt(
  fn: Callable[P, str | Awaitable[str]] | None = None,
  *,
  description: str | None = None,
):
  """Decorator for prompt template functions.

  Usage — simple::

      @prompt
      async def code_review(diff: str, language: str = "python") -> str:
          return f"Review this {language} code:\n```\n{diff}\n```"

      text = await code_review.render(diff="...")

  Usage — with description::

      @prompt(description="Code review prompt")
      def review_summary(count: int) -> str:
          return f"Summarize {count} review comments."

  The decorated function can be sync or async.  Its return value must be
  a string (the prompt text).
  """

  def wrapper(f: Callable[P, str | Awaitable[str]]) -> PromptFunction[P]:
    pf = PromptFunction(f)
    if description:
      pf.__doc__ = description
    return pf

  if fn is None:
    return wrapper
  return wrapper(fn)