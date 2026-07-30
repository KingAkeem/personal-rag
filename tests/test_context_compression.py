import unittest
from dataclasses import dataclass

from context_compression import (
    CompressionConfig,
    build_retrieval_query,
    format_retrieved_context,
)


@dataclass
class Chunk:
    filename: str
    content: str


class ContextCompressionTests(unittest.TestCase):
    def test_disabled_mode_preserves_current_retrieval_and_context(self):
        config = CompressionConfig(enabled=False)
        self.assertEqual(build_retrieval_query("question", [["old", "answer"]], config), "question")
        self.assertEqual(
            format_retrieved_context([Chunk("notes.txt", "full content")], config),
            "From notes.txt:\nfull content",
        )

    def test_enabled_mode_uses_recent_history_and_bounds_context(self):
        config = CompressionConfig(
            enabled=True,
            max_history_chars=40,
            max_chunk_chars=20,
            max_context_chars=45,
        )
        query = build_retrieval_query(
            "What was the code?",
            [["Discuss the train reservation", "It was the Montreal trip"]],
            config,
        )
        self.assertIn("Current question", query)
        self.assertIn("Montreal", query)

        context = format_retrieved_context(
            [Chunk("travel.txt", "one two three four five six seven eight nine")],
            config,
        )
        self.assertLessEqual(len(context), 45)
        self.assertTrue(context.startswith("From travel.txt"))

    def test_current_empty_history_entry_is_not_duplicated(self):
        config = CompressionConfig(enabled=True)
        query = build_retrieval_query("new question", [["new question", ""]], config)
        self.assertEqual(query, "new question")


if __name__ == "__main__":
    unittest.main()
