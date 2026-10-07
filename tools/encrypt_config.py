#!/usr/bin/env python3
"""Encrypt JSON from stdin into a private Container Apps settings file."""

import argparse
import base64
import json
import os
import sys

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt(plaintext):
    json.loads(plaintext)
    key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, b"claude-proxy-config-v1")
    envelope = "v1." + base64.b64encode(nonce + ciphertext).decode("ascii")
    if len(envelope) > 1024 * 1024:
        raise ValueError("Configuration too large")
    return {
        "CLAUDE_PROXY_ENCRYPTED_CONFIG": envelope,
        "CLAUDE_PROXY_CONFIG_KEY": base64.b64encode(key).decode("ascii"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New private file, never overwritten")
    args = parser.parse_args()
    try:
        settings = encrypt(sys.stdin.buffer.read(1024 * 1024 + 1))
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            json.dump(settings, output)
    except (OSError, ValueError):
        sys.exit("Cannot encrypt configuration or create output file")


if __name__ == "__main__":
    main()
