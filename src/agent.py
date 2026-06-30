"""
Core RAG-over-codebase agent (Pydantic AI).

The agent answers questions about a single code repository by retrieving
relevant chunks, reading full files for context, and grepping for callers /
definitions the chunks miss, then returns a STRUCTURED answer with verifiable
citations (file + line range). The structured output is what the eval harness
scores against, so keep the schema stable.

API NOTE — Pydantic AI's surface has shifted across versions. As of writing,
the current names are `output_type` (older versions called it `result_type`)
and `result.output` (older: `result.data`). Pin a version in pyproject.toml
and check the docs if these don't match what you installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from tools import VectorStore, CodeRepo  # infra abstractions (see tools.py)


# --------------------------------------------------------------------------
# Structured output — this is the contract the eval harness scores.
# --------------------------------------------------------------------------

class Citation(BaseModel):
    file_path: str = Field(description="Repo-relative path, e.g. 'src/pipeline.py'")
    start_line: int
    end_line: int


class Answer(BaseModel):
    text: str = Field(description="Answer grounded ONLY in retrieved/read context.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Every file/line range that supports the answer.",
    )


# --------------------------------------------------------------------------
# Dependencies injected per-run. Typed => testable and swappable.
# --------------------------------------------------------------------------

@dataclass
class Deps:
    store: VectorStore   # semantic retrieval over chunked code
    repo: CodeRepo       # raw file read + grep over the working tree


SYSTEM_PROMPT = """You are a code-comprehension assistant for ONE repository.
Answer strictly from information you retrieve or read from the repo.

Workflow:
1. Call `retrieve` to find candidate chunks for the question.
2. Call `read_file` on a chunk's file/line range before relying on it — chunks
   are snippets and often omit the surrounding function or a config constant.
3. Call `grep` to find callers or definitions the chunks did not include.

Rules:
- Never invent file paths, function names, or line numbers.
- If the repo does not support an answer, say so explicitly.
- Every factual claim must map to a citation with a real file and line range.
"""

# The model string is provider-agnostic: swap "openai:gpt-4o" for an Anthropic
# or local model without changing anything else below. Bump PROMPT_VERSION
# whenever you edit the system prompt — the eval harness keys runs on it.
PROMPT_VERSION = "v1"

agent = Agent(
    "openai:gpt-4o",
    deps_type=Deps,
    output_type=Answer,          # older Pydantic AI: result_type=Answer
    system_prompt=SYSTEM_PROMPT,
    retries=2,
)


# --------------------------------------------------------------------------
# Tools — the agent's hands. Each receives RunContext[Deps] for typed access.
# --------------------------------------------------------------------------

@agent.tool
async def retrieve(ctx: RunContext[Deps], query: str, k: int = 6) -> list[dict]:
    """Semantic search over the chunked codebase. Returns chunks + metadata."""
    hits = await ctx.deps.store.search(query, k=k)
    return [
        {
            "file_path": h.file_path,
            "start_line": h.start_line,
            "end_line": h.end_line,
            "symbol": h.symbol_name,
            "snippet": h.text,
            "score": h.score,
        }
        for h in hits
    ]


@agent.tool
async def read_file(
    ctx: RunContext[Deps],
    file_path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> str:
    """Read an exact line range from a repo file for full context."""
    return await ctx.deps.repo.read(file_path, start_line, end_line)


@agent.tool
async def grep(ctx: RunContext[Deps], pattern: str, max_results: int = 20) -> list[dict]:
    """Find where a symbol or string is defined/used across the repo."""
    return await ctx.deps.repo.grep(pattern, max_results=max_results)


# --------------------------------------------------------------------------
# Convenience runner. The harness can call this, capturing tool calls via
# result.all_messages() to compute retrieval precision/recall.
# --------------------------------------------------------------------------

async def ask(question: str, deps: Deps) -> Answer:
    result = await agent.run(question, deps=deps)
    return result.output         # older Pydantic AI: result.data
