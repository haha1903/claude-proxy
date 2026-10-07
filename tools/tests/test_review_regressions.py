import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copilot_vault as v
from test_cx import cx

class ReviewRegressions(unittest.TestCase):
    def test_default_tls_port_continuation_is_same_origin(self):
        origin='https://fixture.vault.azure.net'
        pages={origin+'/secrets?api-version=7.4':{'value':[],'nextLink':origin+':443/secrets?page=2'},
               origin+':443/secrets?page=2':{'value':[{'id':origin+':443/secrets/copilot-1'}]},
               origin+'/secrets/copilot-1?api-version=7.4':{'value':json.dumps({'api_key':'fixture','github':[{'login':'alice','token':'fixture'}]})}}
        with patch.object(v,'configuration',return_value=(origin,'sub')),patch.object(v,'azure_token',return_value='fixture'),patch.object(v,'get_json',side_effect=lambda url,token:pages[url]):
            self.assertEqual(len(v.load_records()['records']),1)
            for target in [origin+':444/secrets', 'https://user@fixture.vault.azure.net/secrets', 'https://evil.invalid:443/secrets']:
                pages[origin+'/secrets?api-version=7.4']['nextLink']=target
                with self.assertRaises(v.VaultError):v.load_records()

    def test_official_and_status_work_without_local_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);(home/'.codex').mkdir()
            config=home/'.codex/config.toml'
            config.write_text('model_provider = "copilot3"\nmodel = "gpt-6-astra"\n')
            with patch.object(Path,'home',return_value=home),patch.object(cx,'discover',side_effect=cx.ConfigError('Local configuration unavailable')) as discover,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cx.main(['status']),0)
                self.assertEqual(cx.main(['official']),0)
                discover.assert_not_called()
                self.assertEqual(cx.main(['copilot3']),1)
            self.assertIn('model_provider = "openai"',config.read_text())



    def test_invalid_cache_cannot_supply_credentials(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(Path,'home',return_value=Path(temp)):
            path=cx.cache_path();path.parent.mkdir(parents=True)
            for data in [{}, {'schema_version':2,'entries':[]},{'schema_version':1,'entries':{}},{'schema_version':1,'entries':[{'number':True,'api_key':'key'}]},{'schema_version':1,'entries':[{'number':1,'api_key':'bad key'}]},{'schema_version':1,'entries':[{'number':1,'api_key':'key'},{'number':2,'api_key':'key'}]}]:
                path.write_text(json.dumps(data))
                with self.assertRaises(cx.ConfigError):cx.discover()



    def test_missing_vault_configuration_does_not_restore_legacy_identities(self):
        import check_usage as usage
        from unittest.mock import Mock
        loader=Mock();loader.load_records.side_effect=v.VaultError('vault_config_invalid')
        with patch.object(usage,'vault_loader',return_value=loader),patch.object(usage,'load_accounts',side_effect=AssertionError('No legacy fallback')):
            report=usage.collect()
        self.assertFalse(report['ok'])
        self.assertEqual(report['accounts'],[])
        self.assertEqual(report['discovery_errors'],[{'code':'vault_config_invalid'}])
