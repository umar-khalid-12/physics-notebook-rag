"""Regression checks for detailed follow-ups and numerical math rendering."""
import json
import unittest
from unittest.mock import Mock, patch

from answer_book import BookAnswer, build_request, format_answer, retrieval_query
from answer_markdown import normalize_math_markdown


PROBLEM = 'A hydraulic press has piston areas 10 cm² and 100 cm². Lift a car of weight 4000 N.'
HISTORY = [{'role': 'user', 'text': PROBLEM}, {'role': 'assistant', 'text': 'The required force is 400 N.'}]
PASSAGES = [{'id': '172', 'page_start': 140, 'page_end': 142, 'text': 'Pressure is transmitted unchanged.'}]
HYDRAULIC = r'''### Calculation
Pascal's law [source:172].
\[F_s = F_L \times \frac{A_s}{A_L}\]
Substituting the numbers:
$$F_s = 4000\,\text{N} \times \frac{10\,\text{cm}^2}{100\,\text{cm}^2}
= 4000\,\text{N} \times 0.1 = 400\,\text{N}$$ ### Answer
**A force of 400 N** lifts the car 【source:172】.
'''
COMPONENTS = r'''### Step 1: Formula
The components are \(F_x = F\cos\theta\) and \(F_y = F\sin\theta\).
### Step 2: Substitution
\[F_x = 160\,\text{N}\times\cos 60^\circ\]
### Step 3: Calculate the horizontal component
$$F_x = 160\,\text{N}\times 0.5 = 80\,\text{N}$$
### Step 4: Calculate the vertical component
$$F_y = 160\,\text{N}\times 0.8660254 \approx 138.6\,\text{N}$$
### Result
Horizontal: **80 N**. Vertical: **138.6 N**. [source:172]
'''


class NumericalTests(unittest.TestCase):
    def test_hydraulic_equations_do_not_swallow_answer(self):
        result = format_answer({'answerable': True, 'formatted_answer': HYDRAULIC}, PASSAGES)
        self.assertIn('\\text{N}\n$$\n\n### Answer', result['answer'])
        self.assertNotIn('【source:', result['answer'])
        self.assertIn('[PDF pp. 140–142]', result['answer'])
        self.assertEqual(result['sources'], PASSAGES)

    def test_components_use_supported_delimiters(self):
        rendered = normalize_math_markdown(COMPONENTS)
        self.assertIn('$F_x = F\\cos\\theta$', rendered)
        self.assertIn('$$\nF_x = 160', rendered)
        self.assertNotIn(r'\[', rendered)
        self.assertNotIn(r'\(', rendered)
        self.assertEqual(normalize_math_markdown(rendered), rendered)

    def test_code_and_comparison_tables_are_unchanged(self):
        text = 'Use `$$x$$`.\n\n```latex\n\\[x = y\\]\n```\n\n| A | B |\n|---|---|\n| $x$ | $y$ |'
        self.assertEqual(normalize_math_markdown(text), text)

    def test_citations_inside_equation_are_moved_out(self):
        rendered = normalize_math_markdown('$$F=400 [PDF p. 140]$$')
        self.assertEqual(rendered, '$$\nF=400\n$$\n\n[PDF p. 140]')

    def test_followup_retrieves_original_problem_and_supplies_history(self):
        search = Mock()
        search.search.return_value = PASSAGES
        question = 'Explain the previous problem in detail, showing every calculation step.'
        with patch('answer_book.call_groq', return_value={'answerable': True, 'formatted_answer': HYDRAULIC}) as api:
            BookAnswer('test-key', search=search).ask(question, history=HISTORY)
        self.assertIn(PROBLEM, search.search.call_args.args[0])
        payload = json.loads(api.call_args.args[0]['messages'][1]['content'])
        self.assertEqual(payload['question'], question)
        self.assertEqual(payload['conversation'], HISTORY)

    def test_new_numerical_does_not_retrieve_previous_problem(self):
        question = 'Show each step for a force of 160 N at 60 degrees to the x-axis.'
        self.assertEqual(retrieval_query(question, HISTORY), question)

    def test_new_concept_and_repeated_detail_requests(self):
        question = 'Explain speed and velocity in detail.'
        self.assertEqual(retrieval_query(question, HISTORY), question)
        history = HISTORY + [{'role':'user', 'text':'Give me the solution in detail'}]
        self.assertIn(PROBLEM, retrieval_query('Show each calculation step', history))

    def test_error_and_preview_payloads_not_sent_as_context(self):
        history = HISTORY + [{'role':'assistant', 'text':'debug payload', 'is_preview': True}]
        payload = json.loads(build_request('More detail', PASSAGES, history=history)['messages'][1]['content'])
        self.assertEqual(payload['conversation'], HISTORY)

    def test_unicode_unknown_citation_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown source'):
            format_answer({'answerable':True, 'formatted_answer':'Wrong 【source:999】'}, PASSAGES)

    def test_prefixed_statement_id_is_validated(self):
        result = {'answerable':True, 'formatted_answer':'Pressure [source:172]',
                  'statements':[{'text':'Pressure', 'source_ids':['source:172']}]}
        self.assertEqual(format_answer(result, PASSAGES)['sources'], PASSAGES)
        result['statements'][0]['source_ids'] = ['source:999']
        with self.assertRaisesRegex(ValueError, 'unknown source'):
            format_answer(result, PASSAGES)


if __name__ == '__main__':
    unittest.main()
