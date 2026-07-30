# Local embedding and compression benchmarks

The benchmark command measures the embedding and retrieval path entirely on the local machine. It uses an in-memory cosine search over a small annotated corpus, so Elasticsearch, Gmail, and cloud APIs are not required.

## Quick smoke test

```bash
PYTHONPATH=src python3 benchmarks/run.py --provider hash
```

The hash provider verifies that the benchmark pipeline works without model dependencies. It is not an application embedding provider and its retrieval score should not be used to choose a model.

## CPU benchmark

Install the optional local encoder and run MiniLM on CPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
python3 benchmarks/run.py \
  --provider sentence-transformers \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --device cpu \
  --output benchmark-results/minilm-cpu.json
```

The model is downloaded on first use and then runs in process. It does not call an embedding API. Configure the model cache for offline use before disconnecting the machine.

MiniLM is a useful CPU choice when GPU memory is limited, Ollama startup dominates short workflows, or the document collection is small enough that lower-dimensional vectors are preferable. Keep the default Ollama `nomic-embed-text` path when existing 768-dimensional indexes must remain compatible or its retrieval quality is better on representative documents.

## Ollama and GPU comparisons

With the existing Ollama service running:

```bash
python3 benchmarks/run.py \
  --provider ollama \
  --model nomic-embed-text \
  --output benchmark-results/nomic-ollama.json
```

To compare sentence-transformers devices on a host with supported PyTorch acceleration, run the same command with `--device cpu` and then `--device cuda`. Keep the model, dataset, `-k`, and compression limits identical. Device availability is managed by the installed PyTorch build.

## Metrics

The JSON report includes:

- provider load time, including local model initialization
- corpus size, ingest time, and documents per second
- peak Python allocation and process peak resident memory
- mean, p50, and p95 query latency
- retrieval recall at `k` against the fixture's expected document IDs
- baseline and compression-enabled retrieval results
- context characters before and after compression, plus reduction ratio

The included corpus is deliberately small and only catches major regressions. For a decision, create a private local dataset with representative document lengths, terminology, and expected document IDs. Do not commit personal source material or benchmark output containing it.

## Compression comparison

Every benchmark run compares normal retrieval with the opt-in extractive compression path. Tune its local character budgets without invoking an LLM:

```bash
python3 benchmarks/run.py \
  --provider hash \
  --max-history-chars 1200 \
  --max-chunk-chars 800 \
  --max-context-chars 3000
```

Watch both recall and context reduction. A smaller prompt is not an improvement if expected documents stop appearing in the top results.
