"""edit's vision_client.py is a copy of read's (PAVE's one model client); the two must not drift."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READ_COPY = ROOT.parent / "read" / "src" / "pave_read" / "vision_client.py"


class SharedClientTests(unittest.TestCase):
    @unittest.skipUnless(READ_COPY.exists(), "read/ is not beside edit/")
    def test_same_file_as_read(self):
        header, body = (ROOT / "pave_edit" / "vision_client.py").read_text().split("\n", 1)
        self.assertTrue(header.startswith("# Copied from read/"))
        self.assertEqual(body, READ_COPY.read_text())


if __name__ == "__main__":
    unittest.main()
