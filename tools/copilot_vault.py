"""Read Copilot records from Key Vault using the selected local Azure CLI identity."""

import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class VaultError(Exception):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def decode(value):
    return json.loads(value, object_pairs_hook=unique_object)


def source_path():
    return Path.home() / ".config/claude-proxy/key-vault.json"


def configuration(path=None):
    try:
        data = decode((path or source_path()).read_text())
        url, subscription = data["vault_url"], data["subscription"]
        if not re.fullmatch(r"https://[a-zA-Z0-9-]+\.vault\.azure\.net/?", url):
            raise ValueError()
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", subscription):
            raise ValueError()
        return url.rstrip("/"), subscription
    except (OSError, ValueError, TypeError, KeyError):
        raise VaultError("vault_config_invalid") from None


def azure_token(subscription):
    try:
        result = subprocess.run(
            ["az", "account", "get-access-token", "--subscription", subscription,
             "--resource", "https://vault.azure.net", "--output", "json"],
            capture_output=True, text=True, timeout=45,
        )
        if result.returncode:
            raise VaultError("azure_authentication_failed")
        token = decode(result.stdout)["accessToken"]
        if not credential(token):
            raise ValueError()
        return token
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        raise VaultError("azure_authentication_failed") from None


def get_json(url, token):
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=25) as response:
            body = response.read(1024 * 1024 + 1)
            if response.status != 200 or len(body) > 1024 * 1024:
                raise VaultError("vault_response_invalid")
        result = decode(body)
        if not isinstance(result, dict):
            raise ValueError()
        return result
    except urllib.error.HTTPError as error:
        error.close()
        raise VaultError("vault_http_" + str(error.code)) from None
    except (OSError, ValueError):
        raise VaultError("vault_read_failed") from None


def credential(value):
    return isinstance(value, str) and bool(value) and all(33 <= ord(c) <= 126 for c in value)


def enabled(attributes):
    now = time.time()
    return (attributes.get("enabled") is not False
            and attributes.get("nbf", 0) <= now < attributes.get("exp", float("inf")))


def parse_record(name, secret):
    try:
        if not enabled(secret.get("attributes", {})):
            return None
        record = decode(secret["value"])
        if not isinstance(record, dict) or set(record) - {"api_key", "policy", "github"}:
            raise ValueError()
        if record.get("policy", "session_hash") != "session_hash" or not credential(record["api_key"]):
            raise ValueError()
        if not isinstance(record["github"], list) or not record["github"]:
            raise ValueError()
        identities = set()
        for account in record["github"]:
            if not isinstance(account, dict) or set(account) != {"login", "token"}:
                raise ValueError()
            login = account["login"]
            if not isinstance(login, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", login):
                raise ValueError()
            if login.lower() in identities or not credential(account["token"]):
                raise ValueError()
            identities.add(login.lower())
        record.setdefault("policy", "session_hash")
        return {"name": name, "number": int(name[8:]), **record}
    except (ValueError, KeyError, TypeError, AttributeError):
        raise VaultError("vault_record_invalid") from None


def load_records(path=None):
    origin, subscription = configuration(path)
    token = azure_token(subscription)
    next_url = origin + "/secrets?api-version=7.4"
    visited, names, inactive = set(), set(), set()
    try:
        while next_url:
            url = urllib.parse.urlsplit(next_url)
            if (url.scheme + "://" + url.netloc != origin or url.path != "/secrets"
                    or url.fragment or next_url in visited or len(visited) >= 100):
                raise VaultError("vault_continuation_invalid")
            visited.add(next_url)
            page = get_json(next_url, token)
            if not isinstance(page["value"], list):
                raise ValueError()
            for item in page["value"]:
                identifier = urllib.parse.urlsplit(item["id"])
                if identifier.scheme + "://" + identifier.netloc != origin:
                    raise VaultError("vault_secret_origin_invalid")
                name = identifier.path.removeprefix("/secrets/")
                if re.fullmatch(r"copilot-[1-9][0-9]*", name) and int(name[8:]) <= 4294967295:
                    (names if enabled(item.get("attributes", {})) else inactive).add(name)
            next_url = page.get("nextLink")
            if next_url is not None and (not isinstance(next_url, str) or not next_url):
                raise VaultError("vault_continuation_invalid")
        records, errors = [], []
        for name in sorted(names, key=lambda n: int(n[8:])):
            try:
                record = parse_record(name, get_json(origin + "/secrets/" + name + "?api-version=7.4", token))
                if record:
                    records.append(record)
                else:
                    inactive.add(name)
            except VaultError as error:
                errors.append({"name": name, "code": str(error)})
        valid = []
        for record in records:
            if sum(r["api_key"] == record["api_key"] for r in records) != 1:
                errors.append({"name": record["name"], "code": "vault_duplicate_binding"})
            else:
                valid.append(record)
        return {"records": valid, "errors": errors, "inactive": sorted(inactive)}
    except (ValueError, KeyError, TypeError, AttributeError):
        raise VaultError("vault_list_invalid") from None
