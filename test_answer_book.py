"""Offline tests; simulated Groq responses do not measure model accuracy."""
import json
import unittest
from unittest.mock import Mock, patch

import httpx

from answer_book import BookAnswer, UNKNOWN, build_request, call_groq, format_answer, list_models

PASSAGES = [{"id": "7", "text": "Speed is distance travelled per unit time.",
             "page_start": 34, "page_end": 35, "source": "physics9.pdf", "section": "Speed"}]
ANSWER = {
    "answerable": True,
    "formatted_answer": (
        "### Formula\n"
        "$$v = \\frac{s}{t}$$\n"
        "- **Speed** is distance travelled per unit time. [source:7]"
    ),
    "statements": [
        {"text": "Speed measures distance travelled per unit time.", "source_ids": ["7"]}
    ],
}


class AnswerTests(unittest.TestCase):
    @patch("answer_book.time.sleep")
    def test_transient_failure_retries_then_succeeds(self, sleep):
        responses = [httpx.Response(503), httpx.Response(200, json={"choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps(ANSWER)}}]})]
        handler = Mock(side_effect=lambda request: responses.pop(0))
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(call_groq({}, "test-key", "openai/gpt-oss-120b", client), ANSWER)
        self.assertEqual(handler.call_count, 2)
        sleep.assert_called_once()

    @patch("answer_book.time.sleep")
    def test_retries_are_bounded(self, sleep):
        handler = Mock(side_effect=lambda request: httpx.Response(503))
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                call_groq({}, "test-key", "openai/gpt-oss-120b", client)
        self.assertEqual(handler.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    @patch("answer_book.time.sleep")
    def test_authentication_failure_is_not_retried(self, sleep):
        handler = Mock(side_effect=lambda request: httpx.Response(403))
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "403"):
                call_groq({}, "test-key", "openai/gpt-oss-120b", client)
        self.assertEqual(handler.call_count, 1)
        sleep.assert_not_called()

    def test_model_list_filters_and_paginates(self):
        first = Mock(status_code=200)
        first.json.return_value = {"data": [
            {"id": "openai/gpt-oss-120b", "object": "model"},
            {"id": "llama-3.1-8b-instant", "object": "model"},
        ]}
        with patch("httpx.Client") as client:
            get = client.return_value.__enter__.return_value.get
            get.side_effect = [first]
            self.assertEqual(list_models("test-key"), ["llama-3.1-8b-instant", "openai/gpt-oss-120b"])
            self.assertEqual(get.call_count, 1)

    def test_404_gives_actionable_model_diagnostic(self):
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(404)
        )) as client, self.assertRaisesRegex(RuntimeError, "--list-models"):
            call_groq({}, "test-key", "unavailable-model", client)

    def test_page_numbers_come_from_local_metadata(self):
        result = format_answer(ANSWER, PASSAGES)
        self.assertIn("[PDF pp. 34–35]", result["answer"])
        self.assertIn("### Formula", result["answer"])
        self.assertEqual(result["sources"], PASSAGES)

    def test_unknown_source_rejected(self):
        bad = {
            "answerable": True,
            "formatted_answer": "An answer [source:999]",
            "statements": [{"text": "An answer", "source_ids": ["999"]}],
        }
        with self.assertRaisesRegex(ValueError, "unknown source"):
            format_answer(bad, PASSAGES)

    def test_uncited_answer_rejected(self):
        with self.assertRaises(ValueError):
            format_answer(
                {
                    "answerable": True,
                    "formatted_answer": "Speed is distance over time.",
                    "statements": [{"text": "Answer", "source_ids": []}],
                },
                PASSAGES,
            )

    def test_missing_formatted_answer_rejected(self):
        with self.assertRaisesRegex(ValueError, "formatted Markdown"):
            format_answer(
                {"answerable": True, "formatted_answer": "", "statements": ANSWER["statements"]},
                PASSAGES,
            )

    def test_comparison_table_citations_are_rewritten(self):
        result = format_answer(
            {
                "answerable": True,
                "formatted_answer": (
                    "### Key Differences\n"
                    "| Aspect | A | B |\n"
                    "| --- | --- | --- |\n"
                    "| Meaning | one [source:7] | two [source:7] |\n"
                ),
                "statements": ANSWER["statements"],
            },
            PASSAGES,
        )
        self.assertIn("| Aspect | A | B |", result["answer"])
        self.assertIn("[PDF pp. 34–35]", result["answer"])
        self.assertNotIn("[source:7]", result["answer"])

    def test_abstention_discards_generated_content(self):
        result = format_answer({"answerable": False, "statements": ANSWER["statements"]}, PASSAGES)
        self.assertEqual(result, {"answerable": False, "answer": UNKNOWN, "sources": []})

    def test_missing_key_fails_before_loading_search(self):
        with patch("answer_book.BookSearch") as search, self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
            BookAnswer("")
        search.assert_not_called()

    def test_full_flow_with_simulated_groq(self):
        search = Mock()
        search.search.return_value = PASSAGES
        with patch("answer_book.call_groq", return_value=ANSWER) as api:
            result = BookAnswer("test-key", search=search).ask("What is speed?", 3)
        search.search.assert_called_once_with("What is speed?", 3)
        context = json.loads(api.call_args.args[0]["messages"][1]["content"])
        self.assertEqual(context["passages"][0]["source_id"], "7")
        self.assertTrue(result["answerable"])

    def test_no_passages_skips_api(self):
        search = Mock()
        search.search.return_value = []
        with patch("answer_book.call_groq") as api:
            self.assertFalse(BookAnswer("test-key", search=search).ask("World cup?")["answerable"])
        api.assert_not_called()

    def test_http_request_and_response(self):
        def handle(request):
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            self.assertNotIn("test-key", str(request.url))
            self.assertEqual(str(request.url), "https://api.groq.com/openai/v1/chat/completions")
            body = json.loads(request.content)
            self.assertEqual(body["messages"][0]["role"], "system")
            self.assertEqual(body["response_format"], {"type": "json_object"})
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
                "content": "<think>reasoning</think>" + json.dumps(ANSWER)}}]})
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            result = call_groq(build_request("Speed?", PASSAGES), "test-key", "openai/gpt-oss-120b", client)
        self.assertEqual(result, ANSWER)

    def test_blocked_truncated_and_malformed_responses(self):
        bodies = [{"choices": []}, {"choices": [{"finish_reason": "length"}]},
                  {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]}]
        for body in bodies:
            with self.subTest(body=body), httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body)
            )) as client, self.assertRaises(RuntimeError):
                call_groq({}, "test-key", "openai/gpt-oss-120b", client)

    def test_api_errors_do_not_expose_remote_body(self):
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(429, text="secret-from-remote")
        )) as client, self.assertRaisesRegex(RuntimeError, "rate limits|quota") as error:
            call_groq({}, "test-key", "openai/gpt-oss-120b", client)
        self.assertNotIn("secret-from-remote", str(error.exception))


if __name__ == "__main__":
    unittest.main()
