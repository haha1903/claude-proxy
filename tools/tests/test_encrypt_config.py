import base64
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SCRIPT = Path(__file__).resolve().parents[1] / "encrypt_config.py"
spec = importlib.util.spec_from_file_location("encrypt_config", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EncryptionTests(unittest.TestCase):
    def test_roundtrip_and_randomness(self):
        plaintext = b'{"client_api_key":"fixture-secret"}'
        first = module.encrypt(plaintext)
        second = module.encrypt(plaintext)
        self.assertNotEqual(first, second)
        data = base64.b64decode(first["CLAUDE_PROXY_ENCRYPTED_CONFIG"][3:])
        key = base64.b64decode(first["CLAUDE_PROXY_CONFIG_KEY"])
        self.assertEqual(AESGCM(key).decrypt(data[:12], data[12:], b"claude-proxy-config-v1"), plaintext)
        self.assertNotIn("fixture-secret", json.dumps(first))

    def test_invalid_and_oversized_input(self):
        for value in [b"not json", json.dumps({"large": "a" * 1024 * 1024}).encode()]:
            with self.assertRaises(ValueError):
                module.encrypt(value)

    def test_cli_writes_private_file_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "settings.json"
            command = [sys.executable, str(SCRIPT), "--output", str(output)]
            result = subprocess.run(command, input=b"{}", capture_output=True)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b"")
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            original = output.read_bytes()
            result = subprocess.run(command, input=b"{}", capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(original, output.read_bytes())
            result = subprocess.run(command, input=b"fixture-secret-invalid-json", capture_output=True)
            self.assertNotIn(b"fixture-secret", result.stderr)


if __name__ == "__main__":
    unittest.main()
