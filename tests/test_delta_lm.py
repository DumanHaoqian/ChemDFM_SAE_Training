import json
import tempfile
import unittest
from pathlib import Path

from delta_lm import load_eval_texts


class EvalTextsTest(unittest.TestCase):
    def test_missing_eval_texts_file_fails_clearly(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_eval_texts("/tmp/does-not-exist-sae-eval.jsonl", 2)
        self.assertIn("Fixed Delta LM eval text file not found", str(ctx.exception))

    def test_load_eval_texts_records_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "eval.jsonl"
            rows = [{"text": "alpha"}, {"text": "beta"}, {"text": "gamma"}]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            texts, meta = load_eval_texts(path, 2)

        self.assertEqual(texts, ["alpha", "beta"])
        self.assertEqual(meta["n_eval_texts"], 2)
        self.assertEqual(meta["n_eval_seqs_requested"], 2)
        self.assertEqual(meta["eval_texts_source"], "fixed_jsonl")
        self.assertRegex(meta["eval_texts_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
