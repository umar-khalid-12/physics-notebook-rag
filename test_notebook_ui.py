"""Offline UI checks: navigation, conversation reset and question submission."""
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class NotebookUITests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file('app.py').run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        return app

    def test_conversation_sources_and_reset(self):
        app = self.app()
        self.assertEqual([tab.label for tab in app.tabs], ['Your notebook', 'Explore the textbook'])
        app.session_state.messages = [
            {'role': 'user', 'text': 'What is force?'},
            {'role': 'assistant', 'text': 'Force changes motion.', 'elapsed': 1,
             'sources': [{'page_start': 23, 'page_end': 24, 'section': 'Forces <example>',
                          'id': 'sample', 'text': 'An excerpt with <brackets> & symbols.'}]},
        ]
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.chat_message), 2)
        rendered = '\n'.join(item.value for item in app.markdown)
        self.assertIn('Forces &lt;example&gt;', rendered)
        self.assertIn('&lt;brackets&gt; &amp; symbols.', rendered)
        next(button for button in app.button if 'New notebook' in button.label).click().run()
        self.assertEqual(app.session_state.messages, [])
        self.assertEqual(len(app.exception), 0)

    @patch('answer_book.BookAnswer.ask')
    @patch('search_book.BookSearch')
    def test_question_card_submits_and_keeps_sources(self, search, ask):
        ask.return_value = {'answer': 'A cited answer.', 'sources': []}
        app = self.app()
        if app.button(key='prompt_01').disabled:
            self.skipTest('Answer service is not configured in this environment')
        app.button(key='prompt_01').click().run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.messages[0]['text'], 'What is the difference between speed and velocity?')
        self.assertEqual(app.session_state.messages[-1]['text'], 'A cited answer.')
        ask.assert_called_once()


if __name__ == '__main__':
    unittest.main()
