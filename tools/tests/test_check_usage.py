import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import ssl
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch
import urllib.error


SCRIPT = Path(__file__).resolve().parents[1] / "check_usage.py"
SPEC = importlib.util.spec_from_file_location("check_usage", SCRIPT)
usage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(usage)
REVIEW = Path.home() / "tmp/review/copilot-usage"


def config_data():
    return {"accounts": {
        str(index): {"login": login, "github_token": "test_credential_" + str(index)}
        for index, login in enumerate(("haha1903", "xilou_microsoft", "litiansun_microsoft"))
    }}


def response_data(login="haha1903"):
    return {
        "login": login,
        "copilot_plan": "enterprise",
        "access_type_sku": "copilot_enterprise_seat_quota",
        "token_based_billing": True,
        "chat_enabled": True,
        "cli_enabled": True,
        "quota_reset_date_utc": "2026-11-01T00:00:00.000Z",
        "quota_snapshots": {
            "premium_interactions": {
                "percent_remaining": 82.7,
                "credits_used": 171383,
                "quota_remaining": 827994.8,
                "entitlement": 1000000,
                "unlimited": False,
            },
            "chat": {"unlimited": True, "remaining": 0},
            "completions": {"unlimited": True, "remaining": 0},
        },
        "unrelated_secret": "must_not_be_printed",
    }


class CoreTests(unittest.TestCase):
    def setUp(self):
        REVIEW.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=REVIEW)
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / "accounts.json"
        self.config.write_text(json.dumps(config_data()))
        self.accounts = list(config_data()["accounts"].values())

    def assert_error(self, code, func, *args):
        with self.assertRaises(usage.CheckError) as caught:
            func(*args)
        self.assertEqual(caught.exception.details["code"], code)
        return caught.exception.details

    def test_config_load_and_credentials(self):
        self.assertEqual(usage.load_accounts(self.config), self.accounts)
        self.assertEqual(usage.resolve_token(self.accounts, "haha1903"), "test_credential_0")

    def test_config_errors(self):
        self.assert_error("config_missing", usage.load_accounts, self.config.parent / "missing")
        with patch.object(Path, "read_text", side_effect=PermissionError("sensitive_error")):
            self.assert_error("config_unreadable", usage.load_accounts, self.config)
        for contents in ("{", '{"accounts":{},"accounts":{}}',
                         '{"accounts":{"duplicate":{},"duplicate":{}}}'):
            with self.subTest(contents=contents):
                self.config.write_text(contents)
                self.assert_error("config_invalid_json", usage.load_accounts, self.config)
        self.config.write_bytes(b"\xff")
        self.assert_error("config_invalid_json", usage.load_accounts, self.config)
        for data in ([], {}, {"accounts": []}, {"accounts": {"bad": None}}):
            with self.subTest(data=data):
                self.config.write_text(json.dumps(data))
                self.assert_error("config_invalid_accounts", usage.load_accounts, self.config)

    def test_missing_duplicate_and_invalid_credentials(self):
        self.assert_error("account_missing", usage.resolve_token, [], "haha1903")
        self.assert_error("account_duplicate", usage.resolve_token, self.accounts * 2, "haha1903")
        for token in (None, "", 123, "space token", "bad\ntoken", "nonascii-\u00e9"):
            with self.subTest(token=token):
                self.assert_error("credential_missing_or_invalid", usage.resolve_token,
                                  [{"login": "haha1903", "github_token": token}], "haha1903")

    def test_whitelist_counters_and_dst(self):
        data = response_data()
        summary = usage.summarize(data)
        self.assertEqual(summary["quota_snapshots"], data["quota_snapshots"])
        self.assertEqual(summary["quota_reset_date_sydney"], "2026-11-01T11:00:00+11:00")
        self.assertNotIn("unrelated_secret", summary)
        self.assertEqual(summary["warnings"], ["credit_counters_disagree"])
        data["quota_reset_date_utc"] = "2026-06-01T00:00:00Z"
        self.assertEqual(usage.summarize(data)["quota_reset_date_sydney"], "2026-06-01T10:00:00+10:00")

    def test_invalid_times_remain_visible_with_warning(self):
        for reset in ("bad", "2026-11-01T00:00:00"):
            summary = usage.summarize({"quota_reset_date_utc": reset})
            self.assertEqual(summary["quota_reset_date_utc"], reset)
            self.assertNotIn("quota_reset_date_sydney", summary)
            self.assertIn("invalid_reset_time", summary["warnings"])

    def test_invalid_types_never_become_zero(self):
        summary = usage.summarize({
            "copilot_plan": {}, "token_based_billing": None,
            "quota_snapshots": {"premium_interactions": {
                "credits_used": True, "quota_remaining": "0", "entitlement": float("nan"),
                "percent_remaining": float("inf"), "unlimited": 1,
            }},
        })
        self.assertNotIn("copilot_plan", summary)
        self.assertNotIn("token_based_billing", summary)
        self.assertEqual(summary["quota_snapshots"]["premium_interactions"], {})
        self.assertIn("premium_quota_unavailable", summary["warnings"])
        self.assertIn("invalid_field:premium_interactions.quota_remaining", summary["warnings"])

    def test_missing_quota_is_unknown(self):
        summary = usage.summarize({})
        self.assertNotIn("quota_snapshots", summary)
        self.assertEqual(summary["warnings"], ["premium_quota_unavailable"])

    def test_unlimited_zero_is_not_exhausted(self):
        quota = {"unlimited": True, "entitlement": 0, "quota_remaining": 0, "credits_used": 2609.7}
        self.assertEqual(usage.quota_warnings(quota), [])
        with patch.object(usage, "get_json", return_value={
            "login": "haha1903", "quota_snapshots": {"premium_interactions": quota}
        }):
            account = usage.check_account(*usage.ACCOUNTS[0], self.accounts)
        text = usage.human_summary({"accounts": [account], "observed_at": "now", "duration_seconds": 0})
        self.assertIn("API 标记无限额", text)
        self.assertNotIn("耗尽", text)
        self.assertIn("2,609.7 credits", text)
        self.assertIn("未提供", text)

    def test_counter_and_percentage_disagreements(self):
        self.assertEqual(usage.quota_warnings({"credits_used": 17.20052, "quota_remaining": 82.79948,
                                              "entitlement": 100, "percent_remaining": 82.7}), [])
        self.assertEqual(usage.quota_warnings({"quota_remaining": 70, "remaining": 80,
                                              "entitlement": 100, "percent_remaining": 90}),
                         ["remaining_counters_disagree", "percentage_disagrees"])
        self.assertEqual(usage.quota_warnings({"quota_remaining": 70, "remaining": 70,
                                              "entitlement": 0, "percent_remaining": 90}), [])

    def test_matching_identity_skips_fallback(self):
        with patch.object(usage, "get_json", return_value=response_data()) as request:
            account = usage.check_account(*usage.ACCOUNTS[0], self.accounts)
        self.assertTrue(account["ok"])
        self.assertTrue(account["identity_verified"])
        request.assert_called_once_with("test_credential_0", "/copilot_internal/user")

    def test_mismatch_and_present_null_never_fallback(self):
        for login in ("another_account", None, ""):
            with patch.object(usage, "get_json", return_value={"login": login}) as request:
                account = usage.check_account(*usage.ACCOUNTS[0], self.accounts)
            self.assertFalse(account["ok"])
            self.assertEqual(account["error"]["code"], "identity_mismatch")
            self.assertNotIn("quota_snapshots", account)
            request.assert_called_once()

    def test_fallback_identity_check(self):
        data = response_data()
        del data["login"]
        for fallback, success in (({"login": "haha1903"}, True),
                                  ({"login": "wrong"}, False), ({}, False)):
            with patch.object(usage, "get_json", side_effect=[data, fallback]) as request:
                account = usage.check_account(*usage.ACCOUNTS[0], self.accounts)
            self.assertEqual(account["ok"], success)
            self.assertEqual(request.call_args.args, ("test_credential_0", "/user"))

    def test_partial_failure_and_redaction(self):
        def request(token, endpoint):
            if token == "test_credential_1":
                raise usage.CheckError("http_error", endpoint=endpoint, http_status=401)
            data = response_data("haha1903" if token.endswith("0") else "litiansun_microsoft")
            data["access_type_sku"] = "echo: " + token
            return data
        with patch.object(usage, "get_json", side_effect=request):
            report = usage.collect(self.config)
        self.assertFalse(report["ok"])
        self.assertEqual([a["ok"] for a in report["accounts"]], [True, False, True])
        encoded = json.dumps(report)
        for account in self.accounts:
            self.assertNotIn(account["github_token"], encoded)
        self.assertNotIn("must_not_be_printed", encoded)
        self.assertIn("[REDACTED]", encoded)
        self.assertIn("HTTP 401", usage.human_summary(report))
        self.assertIn("credit_counters_disagree", usage.human_summary(report))

    def test_config_failure_reports_all_accounts(self):
        report = usage.collect(self.config.parent / "missing")
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["accounts"]), 3)
        self.assertNotIn("HTTP", usage.human_summary(report))

    def test_duplicate_identity_only_blocks_that_account(self):
        data = config_data()
        data["accounts"]["duplicate"] = data["accounts"]["0"]
        self.config.write_text(json.dumps(data))
        def request(token, endpoint):
            return response_data(self.accounts[int(token[-1])]["login"])
        with patch.object(usage, "get_json", side_effect=request):
            report = usage.collect(self.config)
        self.assertEqual([a["ok"] for a in report["accounts"]], [False, True, True])
        self.assertEqual(report["accounts"][0]["error"]["code"], "account_duplicate")

    def test_main_outputs_json_and_exit_status(self):
        report = usage.collect(self.config.parent / "missing")
        for json_mode in (True, False):
            output = io.StringIO()
            with patch.object(usage, "collect", return_value=report), contextlib.redirect_stdout(output):
                self.assertEqual(usage.main(["--json"] if json_mode else []), 1)
            if json_mode:
                self.assertEqual(json.loads(output.getvalue()), report)
            else:
                self.assertIn("Copilot 3", output.getvalue())
        with patch.object(usage, "collect", return_value={**report, "ok": True}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(usage.main(["--json"]), 0)

    def test_invalid_endpoint_and_timeout_are_fixed(self):
        self.assert_error("invalid_endpoint", usage.get_json, "test_credential", "https://other.example/")
        opener = Mock()
        opener.open.side_effect = TimeoutError("sensitive exception")
        with patch.object(usage.urllib.request, "build_opener", return_value=opener):
            self.assert_error("timeout", usage.get_json, "test_credential", "/user")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 25)
        self.assertEqual(opener.open.call_args.args[0].full_url, "https://api.github.com/user")

    def test_network_exceptions_are_sanitized(self):
        for reason, code in ((TimeoutError("private"), "timeout"),
                             (ssl.SSLError("private"), "tls_error"),
                             (socket.gaierror("private"), "dns_error"),
                             (ConnectionRefusedError("private"), "network_error")):
            opener = Mock()
            opener.open.side_effect = urllib.error.URLError(reason)
            with patch.object(usage.urllib.request, "build_opener", return_value=opener):
                details = self.assert_error(code, usage.get_json, "test_credential", "/user")
            self.assertNotIn("private", json.dumps(details))
        opener.open.side_effect = OSError("private")
        with patch.object(usage.urllib.request, "build_opener", return_value=opener):
            self.assert_error("network_error", usage.get_json, "test_credential", "/user")


class HttpIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.requests.append((self.command, self.path, dict(self.headers)))
                status, body, headers = cls.dispatch(self)
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.origin = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=2)

    def setUp(self):
        type(self).requests = []
        type(self).dispatch = staticmethod(lambda req: (200, json.dumps(response_data()).encode(), {}))
        patcher = patch.object(usage, "API_ORIGIN", self.origin)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_real_http_headers_and_proxy_bypass(self):
        with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1", "HTTP_PROXY": "http://127.0.0.1:1"}):
            self.assertEqual(usage.get_json("test_credential", "/copilot_internal/user")["login"], "haha1903")
        method, path, headers = self.requests[0]
        self.assertEqual((method, path), ("GET", "/copilot_internal/user"))
        headers = {k.lower(): v for k, v in headers.items()}
        self.assertEqual(headers["authorization"], "token test_credential")
        self.assertEqual(headers["accept"], "application/json")
        self.assertEqual(headers["user-agent"], "GithubCopilot/1.96.0")
        self.assertEqual(headers["editor-version"], "vscode/1.96.0")

    def test_redirects_are_never_followed(self):
        for status in (301, 302, 303, 307, 308):
            type(self).requests = []
            type(self).dispatch = staticmethod(lambda req: (status, b"", {"Location": self.origin + "/leak"}))
            with self.assertRaises(usage.CheckError) as caught:
                usage.get_json("test_credential", "/user")
            self.assertEqual(caught.exception.details["code"], "redirect_rejected")
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(self.requests[0][1], "/user")

    def test_status_and_malformed_responses(self):
        for status, body, error in (
            (401, b"private_response", "http_error"),
            (403, b"private_response", "http_error"),
            (500, b"private_response", "http_error"),
            (204, b"", "unexpected_http_status"),
            (200, b"[]", "invalid_response"),
            (200, b"not-json", "invalid_response"),
            (200, b' {"login":"a","login":"b"}', "invalid_response"),
            (200, b"x" * (usage.MAX_RESPONSE_BYTES + 1), "response_too_large"),
        ):
            with self.subTest(status=status, error=error):
                type(self).dispatch = staticmethod(lambda req: (status, body, {}))
                with self.assertRaises(usage.CheckError) as caught:
                    usage.get_json("test_credential", "/user")
                self.assertEqual(caught.exception.details["code"], error)
                self.assertNotIn("private_response", json.dumps(caught.exception.details))

    def test_three_accounts_overlap_and_fallback_uses_same_identity(self):
        barrier = threading.Barrier(3, timeout=3)
        def dispatch(req):
            token = req.headers["Authorization"].removeprefix("token ")
            index = int(token[-1])
            data = response_data(config_data()["accounts"][str(index)]["login"])
            if req.path == "/copilot_internal/user":
                try:
                    barrier.wait()
                except threading.BrokenBarrierError:
                    return 500, b"not_concurrent", {}
                if index == 1:
                    del data["login"]
            return 200, json.dumps(data).encode(), {}
        type(self).dispatch = staticmethod(dispatch)
        with patch.object(usage, "load_accounts", return_value=list(config_data()["accounts"].values())):
            report = usage.collect()
        self.assertTrue(report["ok"])
        self.assertEqual([a["name"] for a in report["accounts"]], ["Copilot 1", "Copilot 2", "Copilot 3"])
        self.assertEqual([a["login"] for a in report["accounts"]],
                         ["haha1903", "xilou_microsoft", "litiansun_microsoft"])
        self.assertEqual(len(self.requests), 4)
        fallback = [req for req in self.requests if req[1] == "/user"]
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0][2]["Authorization"], "token test_credential_1")


if __name__ == "__main__":
    unittest.main()
