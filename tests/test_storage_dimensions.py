import io
import unittest
from contextlib import redirect_stdout

from storage.elastic import ElasticsearchStorage


class FakeIndices:
    def exists(self, index):
        return True

    def get_mapping(self, index):
        return {
            index: {
                "mappings": {
                    "properties": {
                        "content_embedding": {"type": "dense_vector", "dims": 768},
                        "filename_embedding": {"type": "dense_vector", "dims": 768},
                        "combined_embedding": {"type": "dense_vector", "dims": 768},
                    }
                }
            }
        }


class FakeClient:
    indices = FakeIndices()


class StorageDimensionTests(unittest.TestCase):
    def test_existing_index_dimension_mismatch_is_actionable(self):
        storage = ElasticsearchStorage("documents", FakeClient(), embedding_dim=384)
        output = io.StringIO()
        with redirect_stdout(output):
            initialized = storage.initialize()

        self.assertFalse(initialized)
        self.assertIn("uses 768-dimension vectors", output.getvalue())
        self.assertIn("Set INDEX_NAME to a new index", output.getvalue())


if __name__ == "__main__":
    unittest.main()
