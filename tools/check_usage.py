#!/usr/bin/env python3
"""Read-only, concurrent subscription checks for Copilot accounts discovered in Key Vault."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal
import http.client
import json
import math
from pathlib import Path
import socket
import ssl
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo


CONFIG = Path.home() / ".config/claude-proxy/copilot-accounts.json"
API_ORIGIN = "https://api.github.com"
TIMEOUT = 25
MAX_RESPONSE_BYTES = 1024 * 1024
SYDNEY = ZoneInfo("Australia/Sydney")
ACCOUNTS = (
    ("Copilot 1", "haha1903"),
    ("Copilot 2", "xilou_microsoft"),
    ("Copilot 3", "litiansun_microsoft"),
)
STATUS_FIELDS = {
    "copilot_plan": str,
    "access_type_sku": str,
    "chat_enabled": bool,
    "cli_enabled": bool,
    "token_based_billing": bool,
    "quota_reset_date_utc": str,
}
QUOTA_FIELDS = {
    "percent_remaining": (int, float),
    "credits_used": (int, float),
    "quota_remaining": (int, float),
    "remaining": (int, float),
    "entitlement": (int, float),
    "unlimited": bool,
    "has_quota": bool,
    "overage_permitted": bool,
    "overage_count": (int, float),
    "timestamp_utc": str,
}


class CheckError(Exception):
    def __init__(self, code, **details):
        self.details = {"code": code, **details}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def load_accounts(path):
    try:
        data = json.loads(path.read_text(), object_pairs_hook=unique_object)
    except FileNotFoundError:
        raise CheckError("config_missing") from None
    except OSError:
        raise CheckError("config_unreadable") from None
    except (ValueError, UnicodeError):
        raise CheckError("config_invalid_json") from None
    if not isinstance(data, dict) or not isinstance(data.get("accounts"), dict):
        raise CheckError("config_invalid_accounts")
    accounts = list(data["accounts"].values())
    if any(not isinstance(account, dict) for account in accounts):
        raise CheckError("config_invalid_accounts")
    return accounts


def resolve_token(accounts, login):
    matches = [account for account in accounts if account.get("login") == login]
    if len(matches) != 1:
        raise CheckError("account_missing" if not matches else "account_duplicate")
    token = matches[0].get("github_token")
    if not isinstance(token, str) or not token or any(
        ord(char) < 33 or ord(char) > 126 for char in token
    ):
        raise CheckError("credential_missing_or_invalid")
    return token


def get_json(token, endpoint):
    if endpoint not in ("/copilot_internal/user", "/user"):
        raise CheckError("invalid_endpoint")
    request = urllib.request.Request(
        API_ORIGIN + endpoint,
        headers={
            "Authorization": "token " + token,
            "Accept": "application/json",
            "User-Agent": "GithubCopilot/1.96.0",
            "editor-version": "vscode/1.96.0",
        },
        method="GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=TIMEOUT) as response:
            if response.status != 200:
                raise CheckError("unexpected_http_status", http_status=response.status)
            body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise CheckError("response_too_large")
        data = json.loads(body, object_pairs_hook=unique_object)
        if not isinstance(data, dict):
            raise CheckError("invalid_response")
        return data
    except urllib.error.HTTPError as error:
        code = "redirect_rejected" if 300 <= error.code < 400 else "http_error"
        error.close()
        raise CheckError(code, endpoint=endpoint, http_status=error.code) from None
    except (TimeoutError, socket.timeout):
        raise CheckError("timeout", endpoint=endpoint) from None
    except urllib.error.URLError as error:
        reason = error.reason
        code = "network_error"
        if isinstance(reason, (TimeoutError, socket.timeout)):
            code = "timeout"
        elif isinstance(reason, ssl.SSLError):
            code = "tls_error"
        elif isinstance(reason, socket.gaierror):
            code = "dns_error"
        raise CheckError(code, endpoint=endpoint) from None
    except (OSError, http.client.HTTPException):
        raise CheckError("network_error", endpoint=endpoint) from None
    except (ValueError, UnicodeError):
        raise CheckError("invalid_response", endpoint=endpoint) from None


def select_fields(source, fields, warnings, prefix=""):
    result = {}
    for key, allowed in fields.items():
        if key not in source:
            continue
        value = source[key]
        types = allowed if isinstance(allowed, tuple) else (allowed,)
        if type(value) not in types or (type(value) is float and not math.isfinite(value)):
            warnings.append("invalid_field:" + prefix + key)
            continue
        result[key] = value
    return result


def quota_warnings(quota):
    if quota.get("unlimited") is True:
        return []
    warnings = []
    values = {key: Decimal(str(value)) for key, value in quota.items()
              if type(value) in (int, float)}
    if {"credits_used", "quota_remaining", "entitlement"} <= values.keys():
        if values["credits_used"] + values["quota_remaining"] != values["entitlement"]:
            warnings.append("credit_counters_disagree")
    if {"remaining", "quota_remaining"} <= values.keys():
        if values["remaining"] != values["quota_remaining"]:
            warnings.append("remaining_counters_disagree")
    if {"percent_remaining", "quota_remaining", "entitlement"} <= values.keys():
        if values["entitlement"] > 0:
            percent = values["quota_remaining"] / values["entitlement"] * 100
            # The API's one-decimal percentage may be truncated rather than rounded.
            if abs(percent - values["percent_remaining"]) > Decimal("0.100001"):
                warnings.append("percentage_disagrees")
    return warnings


def summarize(data):
    warnings = []
    result = select_fields(data, STATUS_FIELDS, warnings)
    reset = result.get("quota_reset_date_utc")
    if reset is not None:
        try:
            parsed = datetime.fromisoformat(reset.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Timezone required")
            result["quota_reset_date_sydney"] = parsed.astimezone(SYDNEY).isoformat()
        except ValueError:
            warnings.append("invalid_reset_time")
    snapshots = data.get("quota_snapshots")
    quotas = {}
    if isinstance(snapshots, dict):
        for kind in ("premium_interactions", "chat", "completions"):
            if isinstance(snapshots.get(kind), dict):
                quotas[kind] = select_fields(snapshots[kind], QUOTA_FIELDS, warnings, kind + ".")
    if quotas:
        result["quota_snapshots"] = quotas
    premium = quotas.get("premium_interactions")
    if premium:
        warnings.extend(quota_warnings(premium))
    else:
        warnings.append("premium_quota_unavailable")
    result["warnings"] = warnings
    return result


def check_account(name, login, accounts):
    result = {"name": name, "expected_login": login, "ok": False}
    try:
        token = resolve_token(accounts, login)
        data = get_json(token, "/copilot_internal/user")
        actual_login = data["login"] if "login" in data else get_json(token, "/user").get("login")
        if not isinstance(actual_login, str) or actual_login.lower() != login.lower():
            raise CheckError("identity_mismatch")
        result.update(summarize(data), ok=True, login=login, identity_verified=True, http_status=200)
    except CheckError as error:
        result["error"] = error.details
    result["observed_at"] = datetime.now(SYDNEY).isoformat(timespec="seconds")
    return result


def redact_strings(value, tokens):
    if isinstance(value, dict):
        return {key: redact_strings(item, tokens) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_strings(item, tokens) for item in value]
    if isinstance(value, str):
        for token in tokens:
            value = value.replace(token, "[REDACTED]")
    return value


def vault_loader():
    import importlib.util
    path = Path(__file__).resolve().parent / "copilot_vault.py"
    if not path.exists():
        path = Path.home() / ".local/lib/copilot-vault/copilot_vault.py"
    spec = importlib.util.spec_from_file_location("copilot_vault", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_vault():
    started = time.monotonic()
    results, errors, inactive, secrets = [], [], [], []
    try:
        vault = vault_loader()
        data = vault.load_records()
        errors, inactive = data["errors"], data["inactive"]
        groups = {}
        for record in data["records"]:
            secrets.append(record["api_key"])
            for account in record["github"]:
                secrets.append(account["token"])
                key = (account["login"].lower(), account["token"])
                groups.setdefault(key, []).append(record["name"])
        def check(item):
            (login, token), names = item
            labels = ["Copilot " + name[8:] for name in names]
            result = check_account(" / ".join(labels), login, [{"login": login, "github_token": token}])
            result["sources"] = names
            return result
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(check, groups.items()))
    except Exception as error:
        code = str(error) if type(error).__name__ == "VaultError" else "vault_loader_failed"
        errors = [{"code": code}]
    report = {
        "observed_at": datetime.now(SYDNEY).isoformat(timespec="seconds"),
        "timezone": "Australia/Sydney", "source": "key_vault",
        "duration_seconds": round(time.monotonic() - started, 2),
        "ok": not errors and all(r["ok"] for r in results),
        "accounts": results, "discovery_errors": errors, "inactive": inactive,
    }
    return redact_strings(report, secrets)


def collect(path=None):
    if path is None and (Path.home() / ".config/claude-proxy/key-vault.json").exists():
        return collect_vault()
    path = path or CONFIG
    started = time.monotonic()
    try:
        accounts = load_accounts(path)
    except CheckError as error:
        results = [{"name": name, "expected_login": login, "ok": False,
                    "error": error.details} for name, login in ACCOUNTS]
    else:
        with ThreadPoolExecutor(max_workers=len(ACCOUNTS)) as pool:
            futures = [pool.submit(check_account, name, login, accounts) for name, login in ACCOUNTS]
            results = [future.result() for future in futures]
        # Redact credentials even if an allowlisted upstream string echoes one.
        tokens = [account["github_token"] for account in accounts
                  if isinstance(account.get("github_token"), str) and account["github_token"]]
        results = redact_strings(results, tokens)
    return {
        "observed_at": datetime.now(SYDNEY).isoformat(timespec="seconds"),
        "timezone": "Australia/Sydney",
        "duration_seconds": round(time.monotonic() - started, 2),
        "ok": all(result["ok"] for result in results),
        "accounts": results,
    }


def human_summary(report):
    lines = ["观察时间：" + report["observed_at"] + "（Australia/Sydney）"]
    for error in report.get("discovery_errors", []):
        lines.append("Key Vault 查询失败：" + error["code"])
    for account in report["accounts"]:
        if not account["ok"]:
            error = account["error"]
            status = " HTTP " + str(error["http_status"]) if "http_status" in error else ""
            lines.append(account["name"] + "：查询失败，" + error["code"] + status)
            continue
        plan = account.get("copilot_plan", "未提供")
        billing = str(account.get("token_based_billing", "未提供")).lower()
        lines.append(f'{account["name"]}：{plan} | token_based_billing={billing}')
        status = [f"{key}={account[key]}" for key in ("access_type_sku", "chat_enabled", "cli_enabled") if key in account]
        if status:
            lines.append("  " + " | ".join(status))
        quotas = account.get("quota_snapshots", {})
        premium = quotas.get("premium_interactions", {})
        parts = ["API 标记无限额"] if premium.get("unlimited") is True else []
        for key, label, unit in (
            ("percent_remaining", "剩余", "%"),
            ("credits_used", "已用", " credits"),
            ("quota_remaining", "余额", " credits"),
            ("entitlement", "总额度", " credits"),
            ("remaining", "remaining", " credits"),
        ):
            if key in premium:
                parts.append(f"{label} {premium[key]:,}{unit}")
        lines.append("  Premium：" + (" | ".join(parts) or "接口未提供额度"))
        if "quota_reset_date_sydney" in account:
            lines.append("  重置：" + account["quota_reset_date_sydney"] + "（悉尼）")
        unlimited = [kind for kind in ("chat", "completions") if quotas.get(kind, {}).get("unlimited") is True]
        if unlimited:
            lines.append("  " + "/".join(unlimited) + "：API 标记无限额")
        if account["warnings"]:
            lines.append("  注意：" + ", ".join(account["warnings"]) + "；保留 API 原值，不推算余额。")
    lines.append(f'查询耗时：{report["duration_seconds"]} 秒')
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print credential-free JSON for monitoring.")
    args = parser.parse_args(argv)
    report = collect()
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else human_summary(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
