import json
from pathlib import Path
import unittest
from rag_eval import Document, RAGWorkbench
from rag_eval.engine import load_corpus, tokens
from rag_eval.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
ENGINE = RAGWorkbench(load_corpus(ROOT / 'data' / 'documents.json'))
CASES = json.loads((ROOT / 'data' / 'golden.json').read_text(encoding='utf-8'))


class RAGLabTests(unittest.TestCase):
    def test_dataset_runs_and_all_citations_are_verbatim(self):
        report = evaluate(ENGINE, CASES)
        self.assertEqual(report['dataset_size'], 8)
        self.assertEqual(report['verbatim_citation_validity'], 1.0)

    def test_retrieval_ranks_golden_documents(self):
        report = evaluate(ENGINE, CASES)
        self.assertEqual(report['recall_at_k'], 1.0, report['cases'])

    def test_abstains_on_unseen_topic(self):
        result = ENGINE.answer('quantum frog telepathy crystals')
        self.assertTrue(result['abstained'])
        self.assertEqual(result['citations'], [])

    def test_private_support_passage_never_leaks_to_public(self):
        result = ENGINE.answer('support only escalation runbook require', role='public')
        self.assertTrue(all('DOC-SUPPORT' not in x['id'] for x in result['retrieved']))
        self.assertTrue(result['abstained'])

    def test_support_can_access_support_data(self):
        result = ENGINE.answer('support only escalation runbook minimal reproduction', role='support')
        self.assertIn('DOC-SUPPORT', result['retrieved'][0]['document_id'])
        self.assertTrue(ENGINE.verify(result, role='support'))
        self.assertFalse(ENGINE.verify(result, role='public'))

    def test_prompt_injection_data_is_never_executed(self):
        result = ENGINE.answer('untrusted retrieved content injected instruction')
        self.assertNotIn('secrets', result.keys())
        self.assertTrue(ENGINE.verify(result))

    def test_citation_tampering_detected(self):
        response = ENGINE.answer('403 missing OAuth scope')
        response['citations'] = ['DOC-DOES-NOT-EXIST#p1']
        self.assertFalse(ENGINE.verify(response))

    def test_cross_role_citation_does_not_validate(self):
        private = ENGINE.answer('support only escalation runbook', role='support')
        self.assertFalse(ENGINE.verify(private, role='public'))

    def test_duplicate_id_rejected(self):
        d = Document('X', 'Hi', 'public', ('Hello world',))
        with self.assertRaises(ValueError):
            RAGWorkbench([d,d])

    def test_invalid_role_and_k_rejected(self):
        with self.assertRaises(ValueError): ENGINE.search('oauth', role='admin')
        with self.assertRaises(ValueError): ENGINE.search('oauth', top_k=0)

    def test_tokenization_stable(self):
        self.assertEqual(tokens('THE OAuth 2.0 scopes!'), ['oauth', 'scopes'])


if __name__ == '__main__':
    unittest.main()
