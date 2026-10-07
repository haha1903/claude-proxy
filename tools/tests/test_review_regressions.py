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

    def test_official_and_status_work_without_vault(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp);(home/'.codex').mkdir()
            config=home/'.codex/config.toml'
            config.write_text('model_provider = "copilot3"\nmodel = "gpt-6-astra"\n')
            with patch.object(Path,'home',return_value=home),patch.object(cx,'discover',side_effect=cx.ConfigError('Vault unavailable')) as discover,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cx.main(['status']),0)
                self.assertEqual(cx.main(['official']),0)
                discover.assert_not_called()
                self.assertEqual(cx.main(['copilot3']),1)
            self.assertIn('model_provider = "openai"',config.read_text())


    def test_sync_updates_providers_without_changing_defaults(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(Path,'home',return_value=Path(temp)),patch.object(cx,'PROXIES',cx.PROXIES.copy()),patch.object(cx,'ALIASES',cx.ALIASES.copy()),patch.object(cx,'VAULT_KEYS',None):
            home=Path(temp);(home/'.codex').mkdir()
            config=home/'.codex/config.toml'
            config.write_text('model_provider = "copilot3"\nmodel = "custom-model"\nmodel_reasoning_effort = "xhigh"\n[model_providers.copilot-proxy]\nbase_url = "https://fixture.invalid"\nname = "Copilot 1"\nexperimental_bearer_token = "old-client-key"\n')
            result={'records':[{'number':1,'api_key':'key-one'},{'number':3,'api_key':'key-three'},{'number':5,'api_key':'key-five'}],'errors':[]}
            from unittest.mock import Mock
            loader=Mock();loader.load_records.return_value=result
            with patch.object(cx,'vault_loader',return_value=loader),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cx.main(['sync']),0)
                parsed=cx.tomllib.loads(config.read_text())
                self.assertEqual(parsed['model_provider'],'copilot3')
                self.assertEqual(parsed['model'],'custom-model')
                self.assertEqual(parsed['model_reasoning_effort'],'xhigh')
                self.assertEqual(parsed['model_providers']['copilot5']['experimental_bearer_token'],'key-five')
                loader.load_records.reset_mock()
                self.assertEqual(cx.main(['copilot5']),0)
                loader.load_records.assert_not_called()
                self.assertEqual(cx.tomllib.loads(config.read_text())['model_provider'],'copilot5')
                original=cx.cache_path().read_bytes()
                with patch.object(cx.os,'replace',side_effect=OSError('private')):
                    self.assertEqual(cx.main(['sync']),1)
                self.assertEqual(cx.cache_path().read_bytes(),original)

    def test_invalid_cache_cannot_supply_credentials(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(Path,'home',return_value=Path(temp)):
            path=cx.cache_path();path.parent.mkdir(parents=True)
            for data in [{}, {'schema_version':2,'entries':[]},{'schema_version':1,'entries':{}},{'schema_version':1,'entries':[{'number':True,'api_key':'key'}]},{'schema_version':1,'entries':[{'number':1,'api_key':'bad key'}]},{'schema_version':1,'entries':[{'number':1,'api_key':'key'},{'number':2,'api_key':'key'}]}]:
                path.write_text(json.dumps(data))
                with self.assertRaises(cx.ConfigError):cx.discover()


    def test_sync_noop_reports_actual_semantics_and_missing_default(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temp,patch.object(Path,'home',return_value=Path(temp)),patch.object(cx,'PROXIES',cx.PROXIES.copy()),patch.object(cx,'ALIASES',cx.ALIASES.copy()),patch.object(cx,'VAULT_KEYS',None):
            home=Path(temp);(home/'.codex').mkdir()
            config=home/'.codex/config.toml'
            config.write_text('model_provider = "copilot3"\nmodel = "custom-model"\n[model_providers.copilot-proxy]\nbase_url = "https://fixture.invalid"\nname = "Copilot 1"\nexperimental_bearer_token = "old-client-key"\n')
            loader=Mock();loader.load_records.return_value={'records':[{'number':1,'api_key':'key-one'},{'number':3,'api_key':'key-three'}],'errors':[]}
            output,error=io.StringIO(),io.StringIO()
            with patch.object(cx,'vault_loader',return_value=loader),contextlib.redirect_stdout(output),contextlib.redirect_stderr(error):
                self.assertEqual(cx.main(['sync']),0)
                output.seek(0);output.truncate(0)
                self.assertEqual(cx.main(['sync']),0)
                self.assertIn('Defaults unchanged',output.getvalue())
                self.assertNotIn('gpt-6-astra',output.getvalue())
                loader.load_records.return_value['records']=[{'number':1,'api_key':'key-one'}]
                self.assertEqual(cx.main(['sync']),0)
                self.assertIn('copilot3 is absent',error.getvalue())
                self.assertEqual(cx.tomllib.loads(config.read_text())['model_provider'],'copilot3')

    def test_missing_vault_configuration_does_not_restore_legacy_identities(self):
        import check_usage as usage
        from unittest.mock import Mock
        loader=Mock();loader.load_records.side_effect=v.VaultError('vault_config_invalid')
        with patch.object(usage,'vault_loader',return_value=loader),patch.object(usage,'load_accounts',side_effect=AssertionError('No legacy fallback')):
            report=usage.collect()
        self.assertFalse(report['ok'])
        self.assertEqual(report['accounts'],[])
        self.assertEqual(report['discovery_errors'],[{'code':'vault_config_invalid'}])
