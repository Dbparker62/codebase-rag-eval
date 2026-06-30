"""
Infrastructure abstractions behind the agent's tools.

Both are Protocols so you can swap implementations without touching agent.py:
  - VectorStore : Chroma/Qdrant locally; pgvector-on-RDS or OpenSearch on AWS.
  - CodeRepo    : a checked-out working tree you read + grep.

A working LocalRepo (ripgrep-backed) is included. The vector store is left as
a thin reference because the embedding model + chunking are choices you'll tune
against the eval harness — that tuning IS the project.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol


# --------------------------------------------------------------------------
# Shared data type
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None
    text: str
    score: float = 0.0


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------

class VectorStore(Protocol):
    async def search(self, query: str, k: int = 6) -> list[Chunk]: ...
    async def upsert(self, chunks: list[Chunk]) -> None: ...


class CodeRepo(Protocol):
    async def read(self, file_path: str, start: int = 1, end: int | None = None) -> str: ...
    async def grep(self, pattern: str, max_results: int = 20) -> list[dict]: ...


# --------------------------------------------------------------------------
# Reference CodeRepo over a local checkout (uses ripgrep: `rg`)
# --------------------------------------------------------------------------

class LocalRepo:
    def __init__(self, root: str):
        self.root = root.rstrip("/")

    async def read(self, file_path: str, start: int = 1, end: int | None = None) -> str:
        path = f"{self.root}/{file_path}"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        end = end or len(lines)
        body = lines[start - 1:end]                       # 1-indexed, inclusive
        return "".join(f"{i}\t{ln}" for i, ln in enumerate(body, start))

    async def grep(self, pattern: str, max_results: int = 20) -> list[dict]:
        proc = subprocess.run(
            ["rg", "--line-number", "--no-heading", pattern, self.root],
            capture_output=True, text=True,
        )
        out: list[dict] = []
        for line in proc.stdout.splitlines()[:max_results]:
            parts = line.split(":", 2)                     # path:line:match
            if len(parts) == 3:
                out.append({
                    "file_path": parts[0].replace(self.root + "/", ""),
                    "line": int(parts[1]),
                    "match": parts[2],
                })
        return out


# --------------------------------------------------------------------------
# Reference VectorStore (Chroma). Embedding + chunking are yours to tune.
# --------------------------------------------------------------------------

class ChromaStore:
    """Thin wrapper. Swap for pgvector/OpenSearch on AWS with the same API."""

    def __init__(self, collection):
        self.collection = collection                       # a chromadb collection

    async def search(self, query: str, k: int = 6) -> list[Chunk]:
        res = self.collection.query(query_texts=[query], n_results=k)
        chunks: list[Chunk] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunks.append(Chunk(
                file_path=meta["file_path"],
                start_line=meta["start_line"],
                end_line=meta["end_line"],
                symbol_name=meta.get("symbol_name"),
                text=doc,
                score=1.0 - dist,                          # cosine distance -> similarity
            ))
        return chunks

    async def upsert(self, chunks: list[Chunk]) -> None:
        self.collection.upsert(
            ids=[f"{c.file_path}:{c.start_line}-{c.end_line}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "symbol_name": c.symbol_name,
            } for c in chunks],
        )
