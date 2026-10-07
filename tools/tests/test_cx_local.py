import contextlib
import io
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
from test_cx import cx


class LocalConfigurationTests(unittest.TestCase):
    def test_config_toml_is_the_only_provider_source(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(Path, "home", return_value=Path(folder)):
            path = Path(folder) / ".codex/config.toml"
            path.parent.mkdir()
            source = 'model_provider = "copilot3"\nmodel = "keep-this-model"\n'
            for number in [5, 3, 1, 4, 2]:
                source += f'\n[model_providers.copilot{number}]\nname = "Copilot {number}"\nbase_url = "https://example.invalid/{number}"\nexperimental_bearer_token = "fixture-key-{number}"\n'
            path.write_text(source)
            original = tomllib.loads(source)
            self.assertEqual(list(cx.discover(original)), [f"copilot{i}" for i in range(1, 6)])
            side_file = Path(folder) / ".config/claude-proxy/copilot-clients.json"
            side_file.parent.mkdir(parents=True)
            side_file.write_text("invalid obsolete data")
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cx.main(["status"]), 0)
                self.assertEqual(path.read_text(), source)
                self.assertEqual(cx.main(["sync"]), 2)
                self.assertEqual(cx.main(["copilot5"]), 0)
            self.assertIn("Configured proxy: Copilot 3", output.getvalue())
            self.assertNotIn("fixture-key", output.getvalue())
            expected = dict(original, model_provider="copilot5")
            self.assertEqual(tomllib.loads(path.read_text()), expected)
            self.assertEqual(side_file.read_text(), "invalid obsolete data")

    def test_quoted_provider_table_keeps_its_contents(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(Path, "home", return_value=Path(folder)):
            path = Path(folder) / ".codex/config.toml"
            path.parent.mkdir()
            source = '[model_providers."copilot3"] # keep\nbase_url = "https://example.invalid"\nexperimental_bearer_token = "fixture-key"\n'
            path.write_text(source)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cx.main(["copilot3"]), 0)
            self.assertTrue(path.read_text().endswith(source))
