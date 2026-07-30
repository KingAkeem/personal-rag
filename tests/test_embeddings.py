import unittest
from unittest.mock import patch

from embeddings import (
    OllamaEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_provider,
)


class FakeOllama:
    def embeddings(self, model, prompt):
        self.call = (model, prompt)
        return {"embedding": [1, 2, 3]}


class FakeVectors:
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)


class FakeEncoder:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, normalize_embeddings, show_progress_bar):
        return [FakeVectors([len(text), 1, 0]) for text in texts]


class EmbeddingProviderTests(unittest.TestCase):
    def test_ollama_provider_preserves_existing_api(self):
        client = FakeOllama()
        provider = OllamaEmbeddingProvider("fixture-model", 3, client)
        self.assertEqual(provider.embed("hello"), [1.0, 2.0, 3.0])
        self.assertEqual(client.call, ("fixture-model", "hello"))

    def test_sentence_transformer_provider_batches_on_cpu(self):
        provider = SentenceTransformerEmbeddingProvider(
            "fixture-model", device="cpu", encoder=FakeEncoder()
        )
        self.assertEqual(provider.dimension, 3)
        self.assertEqual(provider.embed_many(["a", "abcd"]), [[1.0, 1.0, 0.0], [4.0, 1.0, 0.0]])
        self.assertEqual(provider.describe()["device"], "cpu")

    def test_factory_keeps_ollama_as_default(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = create_embedding_provider()
        self.assertEqual(provider.name, "ollama")
        self.assertEqual(provider.model, "nomic-embed-text")
        self.assertEqual(provider.dimension, 768)

    def test_factory_rejects_unknown_provider(self):
        with self.assertRaisesRegex(ValueError, "Unsupported embedding provider"):
            create_embedding_provider("cloud-api")


if __name__ == "__main__":
    unittest.main()
