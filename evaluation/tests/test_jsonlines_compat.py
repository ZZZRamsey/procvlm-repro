import tempfile
import unittest
from pathlib import Path

class JsonlinesCompatTest(unittest.TestCase):
    def test_roundtrip_and_append(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1] / "runtime_compat"))
        import jsonlines
        path = Path(tempfile.mkdtemp()) / "rows.jsonl"
        with jsonlines.open(path, "w") as writer:
            writer.write({"x": 1})
            writer.write_all([{"x": 2}])
        with jsonlines.open(path, "a") as writer:
            writer.write({"x": 3})
        with jsonlines.open(path, "r") as reader:
            self.assertEqual(list(reader), [{"x": 1}, {"x": 2}, {"x": 3}])

if __name__ == "__main__":
    unittest.main()
