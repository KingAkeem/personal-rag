import json
import unittest
from pathlib import Path

from benchmarks.run import HashEmbeddingProvider, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_hash_smoke_benchmark_reports_required_metrics(self):
        root = Path(__file__).resolve().parents[1]
        dataset = json.loads((root / "benchmarks/sample_corpus.json").read_text(encoding="utf-8"))
        result = run_benchmark(HashEmbeddingProvider(), dataset, k=3)

        self.assertEqual(result["provider"]["device"], "cpu")
        self.assertEqual(result["corpus"]["documents"], 6)
        self.assertIn("seconds", result["ingest"])
        self.assertIn("process_peak_rss_mb", result["ingest"])
        self.assertIn("latency_ms_p95", result["retrieval_baseline"])
        self.assertIn("recall_at_k", result["retrieval_baseline"])
        self.assertGreaterEqual(
            result["retrieval_with_compression"]["context_reduction_ratio"], 0
        )


if __name__ == "__main__":
    unittest.main()
