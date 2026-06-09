import pytest

from nonoka.core.prompt import (
  prompt,
  PromptTemplate,
  PromptFunction,
)


# --------------------------------------------------------------------------- #
# PromptTemplate
# --------------------------------------------------------------------------- #

class TestPromptTemplate:
  def test_from_string_basic(self):
    tmpl = PromptTemplate.from_string("Hello, {{name}}!")
    assert tmpl.render_sync(name="World") == "Hello, World!"

  def test_from_string_with_condition(self):
    tmpl = PromptTemplate.from_string("""
{% if strict %}
Be strict.
{% endif %}
Review: {{code}}
""")
    result = tmpl.render_sync(code="x = 1", strict=True)
    assert "Be strict." in result
    assert "Review: x = 1" in result

    result_no_strict = tmpl.render_sync(code="x = 1", strict=False)
    assert "Be strict." not in result_no_strict

  def test_from_string_with_loop(self):
    tmpl = PromptTemplate.from_string("""
{% for item in items %}
- {{item}}
{% endfor %}
""")
    result = tmpl.render_sync(items=["a", "b", "c"])
    assert "- a" in result
    assert "- b" in result
    assert "- c" in result

  def test_from_file(self, tmp_path):
    f = tmp_path / "test.j2"
    f.write_text("Hello, {{name}}!")
    tmpl = PromptTemplate.from_file(f)
    assert tmpl.render_sync(name="File") == "Hello, File!"

  def test_from_file_not_found(self):
    with pytest.raises(FileNotFoundError):
      PromptTemplate.from_file("/nonexistent/path.j2")

  def test_call_shorthand(self):
    tmpl = PromptTemplate("Hello, {{name}}!")
    assert tmpl(name="Shorthand") == "Hello, Shorthand!"

  def test_partial(self):
    tmpl = PromptTemplate("You are {{role}}.\n\n{{content}}").partial(role="engineer")
    result = tmpl.render_sync(content="Write code.")
    assert "You are engineer." in result
    assert "Write code." in result

  def test_partial_chaining(self):
    tmpl = PromptTemplate("{{a}} {{b}} {{c}}")
    p1 = tmpl.partial(a="1")
    p2 = p1.partial(b="2")
    assert p2.render_sync(c="3") == "1 2 3"

  @pytest.mark.asyncio
  async def test_render_async(self):
    tmpl = PromptTemplate("Hello, {{name}}!")
    result = await tmpl.render(name="Async")
    assert result == "Hello, Async!"

  def test_immutable(self):
    tmpl = PromptTemplate("Hello")
    # frozen dataclass — source is set at init
    assert tmpl.source == "Hello"


# --------------------------------------------------------------------------- #
# @prompt decorator
# --------------------------------------------------------------------------- #

class TestPromptDecorator:
  @pytest.mark.asyncio
  async def test_async_prompt_function(self):
    @prompt
    async def code_review(diff: str, language: str = "python") -> str:
      return f"Review this {language} code:\n```\n{diff}\n```"

    assert isinstance(code_review, PromptFunction)
    result = await code_review.render(diff="x = 1", language="go")
    assert "Review this go code:" in result
    assert "x = 1" in result

  def test_sync_prompt_function(self):
    @prompt
    def greeting(name: str) -> str:
      return f"Hello, {name}!"

    assert isinstance(greeting, PromptFunction)
    result = greeting.render_sync(name="World")
    assert result == "Hello, World!"

  def test_prompt_with_description(self):
    @prompt(description="A greeting prompt")
    def greet(name: str) -> str:
      return f"Hi, {name}"

    assert greet.__doc__ == "A greeting prompt"

  def test_prompt_call(self):
    @prompt
    def simple(name: str) -> str:
      return f"Hello {name}"

    result = simple("test")
    assert result == "Hello test"

  def test_async_prompt_cannot_render_sync(self):
    @prompt
    async def async_only(name: str) -> str:
      return f"Hello {name}"

    with pytest.raises(RuntimeError, match="async"):
      async_only.render_sync(name="test")

  @pytest.mark.asyncio
  async def test_sync_prompt_can_render_async(self):
    @prompt
    def sync_only(name: str) -> str:
      return f"Hello {name}"

    result = await sync_only.render(name="test")
    assert result == "Hello test"
