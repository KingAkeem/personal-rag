#!/usr/bin/env python3
"""Run a local embedding, retrieval, and compression benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from context_compression import (  # noqa: E402
    CompressionConfig,
    build_retrieval_query,
    format_retrieved_context,
)
from embeddings import EmbeddingProvider, create_embedding_provider  # noqa: E402


class HashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free benchmark smoke provider; not intended for application use."""

    name = "hash-smoke"
    model = "blake2b-token-hash"
    device = "cpu"

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class BenchmarkChunk:
    def __init__(self, filename: str, content: str) -> None:
        self.filename = filename
        self.content = content


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[max(0, index)]


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _search(
    query: str,
    provider: EmbeddingProvider,
    documents: Sequence[Dict[str, Any]],
    document_vectors: Sequence[Sequence[float]],
    k: int,
) -> List[Tuple[Dict[str, Any], float]]:
    query_vector = provider.embed(query)
    ranked = sorted(
        zip(documents, document_vectors),
        key=lambda item: _cosine(query_vector, item[1]),
        reverse=True,
    )
    return [(document, _cosine(query_vector, vector)) for document, vector in ranked[:k]]


def _evaluate_queries(
    provider: EmbeddingProvider,
    documents: Sequence[Dict[str, Any]],
    document_vectors: Sequence[Sequence[float]],
    queries: Sequence[Dict[str, Any]],
    k: int,
    compression: CompressionConfig,
) -> Dict[str, Any]:
    latencies: List[float] = []
    hits = 0
    query_chars_before = 0
    query_chars_after = 0
    context_chars_before = 0
    context_chars_after = 0

    for item in queries:
        query = build_retrieval_query(item["text"], item.get("history", []), compression)
        query_chars_before += len(item["text"])
        query_chars_after += len(query)
        started = time.perf_counter()
        ranked = _search(query, provider, documents, document_vectors, k)
        latencies.append((time.perf_counter() - started) * 1000)
        if item["expected"] in {document["id"] for document, _ in ranked}:
            hits += 1

        chunks = [BenchmarkChunk(document["filename"], document["content"]) for document, _ in ranked]
        baseline_context = format_retrieved_context(chunks, CompressionConfig(enabled=False))
        compressed_context = format_retrieved_context(chunks, compression)
        context_chars_before += len(baseline_context)
        context_chars_after += len(compressed_context)

    return {
        "queries": len(queries),
        "recall_at_k": round(hits / len(queries), 4) if queries else 0.0,
        "latency_ms_mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "latency_ms_p50": round(statistics.median(latencies), 3) if latencies else 0.0,
        "latency_ms_p95": round(_percentile(latencies, 0.95), 3),
        "query_chars_before": query_chars_before,
        "query_chars_after": query_chars_after,
        "context_chars_before": context_chars_before,
        "context_chars_after": context_chars_after,
        "context_reduction_ratio": round(
            1 - (context_chars_after / context_chars_before), 4
        ) if context_chars_before else 0.0,
    }


def run_benchmark(
    provider: EmbeddingProvider,
    dataset: Dict[str, Any],
    k: int = 3,
    max_history_chars: int = 800,
    max_chunk_chars: int = 160,
    max_context_chars: int = 400,
) -> Dict[str, Any]:
    documents = dataset["documents"]
    queries = dataset["queries"]

    tracemalloc.start()
    ingest_started = time.perf_counter()
    document_vectors = provider.embed_many([document["content"] for document in documents])
    ingest_seconds = time.perf_counter() - ingest_started
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    baseline = _evaluate_queries(
        provider,
        documents,
        document_vectors,
        queries,
        k,
        CompressionConfig(enabled=False),
    )
    compressed = _evaluate_queries(
        provider,
        documents,
        document_vectors,
        queries,
        k,
        CompressionConfig(
            enabled=True,
            max_history_chars=max_history_chars,
            max_chunk_chars=max_chunk_chars,
            max_context_chars=max_context_chars,
        ),
    )
    return {
        "provider": provider.describe(),
        "corpus": {
            "documents": len(documents),
            "characters": sum(len(document["content"]) for document in documents),
        },
        "ingest": {
            "seconds": round(ingest_seconds, 4),
            "documents_per_second": round(len(documents) / ingest_seconds, 2)
            if ingest_seconds else 0.0,
            "peak_python_memory_mb": round(peak_python_bytes / (1024 * 1024), 3),
            "process_peak_rss_mb": round(_peak_rss_mb(), 3),
        },
        "retrieval_baseline": baseline,
        "retrieval_with_compression": compressed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("hash", "ollama", "sentence-transformers"),
        default="hash",
        help="hash is a dependency-free smoke test; use a model provider for meaningful results",
    )
    parser.add_argument("--model", help="provider model override")
    parser.add_argument("--device", default="cpu", help="sentence-transformers device")
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/sample_corpus.json")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument("-k", type=int, default=3, help="retrieval result count")
    parser.add_argument("--max-history-chars", type=int, default=800)
    parser.add_argument("--max-chunk-chars", type=int, default=160)
    parser.add_argument("--max-context-chars", type=int, default=400)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.k < 1:
        raise SystemExit("-k must be at least 1")

    provider_started = time.perf_counter()
    if args.provider == "hash":
        provider: EmbeddingProvider = HashEmbeddingProvider()
    else:
        provider = create_embedding_provider(args.provider, args.model, args.device)
    provider_load_seconds = time.perf_counter() - provider_started

    with args.dataset.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    result = run_benchmark(
        provider,
        dataset,
        k=args.k,
        max_history_chars=args.max_history_chars,
        max_chunk_chars=args.max_chunk_chars,
        max_context_chars=args.max_context_chars,
    )
    result["provider_load_seconds"] = round(provider_load_seconds, 4)
    result["dataset"] = os.fspath(args.dataset)

    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
