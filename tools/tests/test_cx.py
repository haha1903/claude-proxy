import contextlib
import copy
import importlib.machinery
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import tomllib
import unittest
from unittest.mock import patch

SCRIPT = str(Path(__file__).resolve().parents[1] / 'cx')
loader = importlib.machinery.SourceFileLoader('cx', SCRIPT)
spec = importlib.util.spec_from_loader(loader.name, loader)
cx = importlib.util.module_from_spec(spec)
loader.exec_module(cx)


class TerminalInput(io.StringIO):
    def isatty(self):
        return True


class CxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.object(Path, 'home', return_value=self.root)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.temp.cleanup)
        self.config = self.root / '.codex/config.toml'
        self.config.parent.mkdir()
        self.key1 = 'fixture-existing-key'
        self.key2 = 'fixture-new-key-"\\$'
        self.key3 = 'fixture-third-key'
        self.config.write_text('''# Keep this comment
model = "gpt-6-astra"
model_provider = "copilot-proxy"
model_reasoning_effort = "xhigh"

[model_providers.copilot-proxy]
name = "Copilot 1"
base_url = "https://example.invalid"
experimental_bearer_token = "fixture-existing-key"

# Keep this table
[projects."/example"]
trust_level = "trusted"
''')
        with self.config.open('a') as handle:
            for number, key in enumerate([self.key1, self.key2, self.key3], 1):
                handle.write(f'\n[model_providers.copilot{number}]\nname = "Copilot {number}"\nbase_url = "https://example.invalid"\nexperimental_bearer_token = {json.dumps(key)}\n')
        self.original = self.config.read_bytes()
        self.parsed = self.read_config()


    def read_config(self):
        return tomllib.loads(self.config.read_text())

    def invoke(self, args, terminal=None):
        output, error = io.StringIO(), io.StringIO()
        stdin = io.StringIO() if terminal is None else TerminalInput(terminal)
        with patch('sys.stdin', stdin), contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            result = cx.main(args)
        text = output.getvalue() + error.getvalue()
        for secret in [self.key1, self.key2, self.key3, 'private-login-one', 'private-login-two', 'private-login-three',
                       'haha1903', 'xilou_microsoft', 'litiansun_microsoft']:
            self.assertNotIn(secret, text)
        return result, text

    def assert_switched(self, key, provider=None):
        expected = copy.deepcopy(self.parsed)
        selection = {self.key1: 'copilot1', self.key2: 'copilot2', self.key3: 'copilot3'}[key]
        expected.update(model_provider=provider or selection, model='gpt-6-astra')
        self.assertEqual(self.read_config(), expected)
        self.assertIn('# Keep this comment', self.config.read_text())
        self.assertIn('# Keep this table', self.config.read_text())

    def test_status_and_noninteractive_default_do_not_write(self):
        for args in [[], ['status']]:
            code, output = self.invoke(args)
            self.assertEqual(code, 0)
            self.assertIn('Configured proxy: Copilot 1', output)
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertFalse(list(self.config.parent.glob('*.backup-*')))

    def test_direct_switch_both_ways_backup_permissions_and_idempotence(self):
        self.assertEqual(self.invoke(['copilot2'])[0], 0)
        self.assert_switched(self.key2)
        self.assertEqual(stat.S_IMODE(self.config.stat().st_mode), 0o600)
        backups = list(self.config.parent.glob('*.backup-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), self.original)
        self.assertEqual(stat.S_IMODE(backups[0].stat().st_mode), 0o600)
        self.assertIn('Already using', self.invoke(['copilot2'])[1])
        self.assertEqual(len(list(self.config.parent.glob('*.backup-*'))), 1)
        self.assertIn('Configured proxy: Copilot 2', self.invoke(['status'])[1])
        self.assertEqual(self.invoke(['copilot1'])[0], 0)
        self.assert_switched(self.key1)

    def test_menu_switches_and_reprompts_invalid_selection(self):
        code, output = self.invoke([], terminal='9\n3\n')
        self.assertEqual(code, 0)
        self.assertIn('1) Official OpenAI', output)
        self.assertIn('2) Copilot 1', output)
        self.assertIn('3) Copilot 2', output)
        self.assertIn('4) Copilot 3', output)
        self.assertIn('5) Status', output)
        self.assertIn('Enter 0, 1, 2, 3, 4 or 5.', output)
        self.assert_switched(self.key2)
        self.assertEqual(self.invoke([], terminal='2\n')[0], 0)
        self.assert_switched(self.key1)
        self.assertEqual(self.invoke([], terminal='1\n')[0], 0)
        self.assert_switched(self.key1, 'openai')


    def test_menu_status_cancel_blank_eof_and_interrupt_do_not_write(self):
        for text in ['5\n', '0\n', '\n', '']:
            self.assertEqual(self.invoke([], terminal=text)[0], 0)
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            self.assertEqual(self.invoke([], terminal='')[0], 0)
        self.assertEqual(self.config.read_bytes(), self.original)

    def test_legacy_copilot_command_uses_copilot1(self):
        self.assertEqual(self.invoke(['copilot', 'copilot2'])[0], 0)
        self.assertEqual(self.invoke(['official'])[0], 0)
        self.assert_switched(self.key2, 'openai')
        self.assertEqual(self.invoke(['copilot'])[0], 0)
        self.assert_switched(self.key1)

    def test_legacy_proxy_shortcuts_remain_compatible(self):
        self.assertEqual(self.invoke(['proxy2'])[0], 0)
        self.assert_switched(self.key2)
        self.assertEqual(self.invoke(['proxy1'])[0], 0)
        self.assert_switched(self.key1)

    def test_third_proxy_direct_alias_legacy_and_menu_switch(self):
        for args, terminal in [(['copilot3'], None), (['proxy3'], None),
                               (['copilot', 'copilot3'], None), ([], '4\n')]:
            with self.subTest(args=args, terminal=terminal):
                self.config.write_bytes(self.original)
                self.assertEqual(self.invoke(args, terminal=terminal)[0], 0)
                self.assert_switched(self.key3)
                self.assertIn('Configured proxy: Copilot 3', self.invoke(['status'])[1])
                self.assertIn('Already using', self.invoke(['copilot3'])[1])


    def test_usage_and_bad_arguments_do_not_write(self):
        for args in [['--help'], ['-h']]:
            self.assertEqual(self.invoke(args)[0], 0)
        for args in [['unknown'], ['status', 'extra'], ['copilot', 'unknown'], ['copilot1', 'extra']]:
            self.assertEqual(self.invoke(args)[0], 2)
        self.assertEqual(self.config.read_bytes(), self.original)


    def test_missing_config_and_invalid_toml(self):
        self.config.unlink()
        self.assertEqual(self.invoke(['status'])[0], 1)
        self.config.write_text('model = [')
        self.assertEqual(self.invoke(['status'])[0], 1)

    def test_missing_provider_and_insert_root_settings(self):
        self.config.write_text('# Empty settings\n')
        self.assertEqual(self.invoke(['copilot1'])[0], 1)
        self.assertEqual(self.invoke(['official'])[0], 0)
        self.assertEqual(self.read_config(), {'model_provider': 'openai'})


    def test_unrecognized_table_layout_fails_closed(self):
        self.config.write_text('[model_providers]\ncopilot-proxy = { base_url = "https://example.invalid" }\n')
        before = self.config.read_bytes()
        self.assertEqual(self.invoke(['copilot2'])[0], 1)
        self.assertEqual(self.config.read_bytes(), before)

    def test_validation_rejects_changes_inside_multiline_strings(self):
        self.config.write_text('description = """\nmodel_provider = not a setting\n"""\n' + self.original.decode())
        before = self.config.read_bytes()
        self.assertEqual(self.invoke(['copilot2'])[0], 1)
        self.assertEqual(self.config.read_bytes(), before)

    def test_concurrent_config_edit_is_preserved(self):
        newer = self.original + b'\n# New external edit\n'
        def choose(config):
            self.config.write_bytes(newer)
            return 'copilot2'
        with patch.object(cx, 'choose_mode', side_effect=choose):
            self.assertEqual(self.invoke([], terminal='3\n')[0], 1)
        self.assertEqual(self.config.read_bytes(), newer)
        self.assertFalse(list(self.config.parent.glob('*.backup-*')))

    def test_replace_failure_preserves_config_and_cleans_temp(self):
        with patch.object(cx.os, 'replace', side_effect=PermissionError):
            self.assertEqual(self.invoke(['copilot2'])[0], 1)
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertFalse(list(self.config.parent.glob('.config-switch-*')))

    def test_default_switch_preserves_every_provider_binding(self):
        self.assertEqual(self.invoke(['copilot3'])[0], 0)
        configured = self.read_config()
        self.assertEqual(configured['model_provider'], 'copilot3')
        providers = copy.deepcopy(configured['model_providers'])
        self.assertEqual(providers['copilot-proxy']['experimental_bearer_token'], self.key1)
        for selection, key in [('copilot1', self.key1), ('copilot2', self.key2), ('copilot3', self.key3)]:
            self.assertEqual(providers[selection]['experimental_bearer_token'], key)
        for selection in ['copilot1', 'copilot2', 'official', 'copilot3']:
            self.assertEqual(self.invoke([selection])[0], 0)
            self.assertEqual(self.read_config()['model_providers'], providers)


if __name__ == '__main__':
    unittest.main()
