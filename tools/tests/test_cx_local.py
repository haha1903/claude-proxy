import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_cx import cx


class LocalConfigurationTests(unittest.TestCase):
    def test_five_fixed_entries_are_read_without_sync(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(Path, "home", return_value=Path(folder)), patch.object(cx, "PROXIES", cx.PROXIES.copy()), patch.object(cx, "ALIASES", cx.ALIASES.copy()), patch.object(cx, "LOCAL_KEYS", None):
            path = cx.cache_path()
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"schema_version": 1, "entries": [
                {"number": number, "api_key": f"fixture-key-{number}"} for number in range(1, 6)
            ]}))
            before = path.read_bytes()
            self.assertEqual(cx.discover()["copilot5"], "fixture-key-5")
            self.assertEqual(cx.PROXIES["copilot5"][0], "Copilot 5")
            self.assertEqual(cx.ALIASES["proxy5"], "copilot5")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cx.main(["sync"]), 2)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(hasattr(cx, "sync_cache"))
            self.assertFalse(hasattr(cx, "vault_loader"))
