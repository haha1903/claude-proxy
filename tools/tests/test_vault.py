import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch, Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copilot_vault as v
import check_usage as usage
from test_cx import cx


def record(key='client-fixture', login='alice', token='github-fixture'):
    return {'api_key':key,'github':[{'login':login,'token':token}]}

class VaultTests(unittest.TestCase):
    def test_schema_policy_duplicate_fields_and_activation(self):
        for value in [record(), {**record(), 'policy':'session_hash'}]:
            parsed=v.parse_record('copilot-1',{'value':json.dumps(value)})
            self.assertEqual(parsed['policy'],'session_hash')
        for value in [[], {}, {**record(),'policy':'most_remaining'}, {**record(),'unexpected':1},
                      {**record(),'api_key':'bad key'}, {**record(),'github':[]},
                      {**record(),'github':[{'login':'alice','token':'x'},{'login':'ALICE','token':'y'}]},
                      record(login='bad login'),record(token='bad token'),{**record(),'github':[{}]}]:
            with self.assertRaises(v.VaultError): v.parse_record('copilot-1',{'value':json.dumps(value)})
        self.assertIsNone(v.parse_record('copilot-1',{'attributes':{'enabled':False}}))
        self.assertFalse(v.enabled({'exp':0}))
        self.assertFalse(v.enabled({'nbf':1e20}))
        with self.assertRaises(ValueError): v.decode('{"key":1,"key":2}')

    def test_config_and_azure_auth_failures_are_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'config.json'
            for value in [{}, {'vault_url':'https://bad.invalid','subscription':'x'},
                          {'vault_url':'https://fixture.vault.azure.net/','subscription':'4496e94c-b276-44d5-8809-1233f334e678'}]:
                path.write_text(json.dumps(value))
                if 'subscription' in value and len(value['subscription'])==36:
                    self.assertEqual(v.configuration(path)[0],'https://fixture.vault.azure.net')
                else:
                    with self.assertRaises(v.VaultError): v.configuration(path)
        for result in [subprocess.CompletedProcess([],1,'','private'),subprocess.CompletedProcess([],0,'bad',''),subprocess.CompletedProcess([],0,'{"accessToken":"bad token"}','')]:
            with patch.object(v.subprocess,'run',return_value=result):
                with self.assertRaises(v.VaultError) as caught: v.azure_token('subscription')
                self.assertNotIn('private',str(caught.exception))
        with patch.object(v.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'{"accessToken":"fixture"}','')) as run:
            self.assertEqual(v.azure_token('subscription'),'fixture')
            self.assertIn('--subscription',run.call_args.args[0])
        with patch.object(v.subprocess,'run',side_effect=OSError('private')):
            with self.assertRaises(v.VaultError): v.azure_token('subscription')

    def test_discovery_pagination_and_partial_failures(self):
        origin='https://fixture.vault.azure.net'
        pages={origin+'/secrets?api-version=7.4':{'value':[{'id':origin+'/secrets/copilot-1'},{'id':origin+'/secrets/req-unrelated'},{'id':origin+'/secrets/copilot-4','attributes':{'enabled':False}}],'nextLink':origin+'/secrets?page=2'},
               origin+'/secrets?page=2':{'value':[{'id':origin+'/secrets/copilot-2'},{'id':origin+'/secrets/copilot-3'}]},
               origin+'/secrets/copilot-1?api-version=7.4':{'value':json.dumps(record())},
               origin+'/secrets/copilot-2?api-version=7.4':{'value':json.dumps(record('another-key'))},
               origin+'/secrets/copilot-3?api-version=7.4':{'value':'bad'}}
        with patch.object(v,'configuration',return_value=(origin,'sub')),patch.object(v,'azure_token',return_value='fixture'),patch.object(v,'get_json',side_effect=lambda url,token:pages[url]) as get:
            data=v.load_records()
            self.assertEqual(len(data['records']),2)
            self.assertEqual(data['inactive'],['copilot-4'])
            self.assertEqual(data['errors'],[{'name':'copilot-3','code':'vault_record_invalid'}])
            self.assertFalse(any('/secrets/req-unrelated' in call.args[0] for call in get.call_args_list))
            pages[origin+'/secrets/copilot-2?api-version=7.4']['value']=json.dumps(record())
            self.assertEqual(len(v.load_records()['records']),0)
            for bad in [origin+'/secrets?api-version=7.4','https://evil.invalid/secrets',origin+'/wrong',False]:
                pages[origin+'/secrets?api-version=7.4']['nextLink']=bad
                with self.assertRaises(v.VaultError):v.load_records()
            pages[origin+'/secrets?api-version=7.4']={'value':[{'id':'https://evil.invalid/secrets/copilot-1'}]}
            with self.assertRaises(v.VaultError):v.load_records()

class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.calls+=1
                self.send_response(cls.status)
                self.send_header('Location',cls.origin+'/leak')
                self.end_headers()
                self.wfile.write(cls.body)
            def log_message(self,*args):pass
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.origin='http://127.0.0.1:'+str(cls.server.server_port)
        cls.worker=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.worker.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.worker.join(2)
    def test_http_errors_redirects_invalid_json_and_size(self):
        for status,body in [(200,b'{}'),(302,b'{}'),(401,b'private'),(200,b'[]'),(200,b'bad'),(200,b'x'*(1024*1024+1))]:
            type(self).status=status;type(self).body=body;type(self).calls=0
            if status==200 and body==b'{}':self.assertEqual(v.get_json(self.origin,'fixture'),{})
            else:
                with self.assertRaises(v.VaultError):v.get_json(self.origin,'fixture')
            self.assertEqual(self.calls,1)

class ConsumerTests(unittest.TestCase):
    def setUp(self):
        self.loader=Mock()
        self.loader.load_records.return_value={'records':[{'name':'copilot-1','number':1,**record()},{'name':'copilot-5','number':5,**record('pool-key')}],'errors':[],'inactive':[]}
    def test_monitor_deduplicates_overlapping_pools_and_redacts_all_secrets(self):
        with patch.object(usage,'vault_loader',return_value=self.loader),patch.object(usage,'get_json',return_value={'login':'alice','copilot_plan':'echo client-fixture github-fixture pool-key'}) as get:
            report=usage.collect_vault()
        self.assertTrue(report['ok'])
        self.assertEqual(len(report['accounts']),1)
        self.assertEqual(report['accounts'][0]['sources'],['copilot-1','copilot-5'])
        self.assertEqual(get.call_count,1)
        for key in ['client-fixture','github-fixture','pool-key']:self.assertNotIn(key,json.dumps(report))
    def test_monitor_discovery_failure_does_not_fallback(self):
        self.loader.load_records.side_effect=v.VaultError('vault_http_403')
        with patch.object(usage,'vault_loader',return_value=self.loader):report=usage.collect_vault()
        self.assertFalse(report['ok']);self.assertEqual(report['accounts'],[])
        self.assertEqual(report['discovery_errors'],[{'code':'vault_http_403'}])
        self.assertIn('vault_http_403',usage.human_summary(report))

if __name__=='__main__':unittest.main()
