import sys
import types
import unittest
from dataclasses import dataclass


generated_prompts = []


def fake_generate(model, prompt, stream):
    generated_prompts.append(prompt)
    return iter([{"response": "answer"}])


sys.modules["ollama"] = types.SimpleNamespace(generate=fake_generate)

from context_compression import CompressionConfig
from llm import rag_chat


@dataclass
class Chunk:
    filename: str
    content: str
    score: float = 1.0


class RagChatTests(unittest.TestCase):
    def setUp(self):
        generated_prompts.clear()

    def test_default_path_retrieves_with_current_message_only(self):
        seen_queries = []

        def search(query, get_embedding, k):
            seen_queries.append(query)
            return [Chunk("notes.txt", "complete source content")]

        output = list(
            rag_chat(
                "current question",
                [["old question", "old answer"]],
                3,
                search,
                lambda text: [1.0],
                CompressionConfig(enabled=False),
            )
        )

        self.assertEqual(seen_queries, ["current question"])
        self.assertIn("complete source content", generated_prompts[0])
        self.assertIn("answer", output[-1])

    def test_compression_path_adds_history_and_caps_context(self):
        seen_queries = []

        def search(query, get_embedding, k):
            seen_queries.append(query)
            return [Chunk("notes.txt", "word " * 100)]

        list(
            rag_chat(
                "What was the code?",
                [["Which trip?", "The Montreal train trip."]],
                3,
                search,
                lambda text: [1.0],
                CompressionConfig(
                    enabled=True,
                    max_history_chars=100,
                    max_chunk_chars=40,
                    max_context_chars=80,
                ),
            )
        )

        self.assertIn("Montreal", seen_queries[0])
        context = generated_prompts[0].split("CONTEXT FROM USER'S DOCUMENTS:\n", 1)[1]
        context = context.split("\n\nUSER'S QUESTION", 1)[0]
        self.assertLessEqual(len(context), 80)


if __name__ == "__main__":
    unittest.main()
