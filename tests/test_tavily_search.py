"""Unit tests for Tavily result normalization without network access or an API key."""

import unittest

from src.web_research.tavily_search import MAX_CONTENT_CHARS_PER_SOURCE, normalize_tavily_results


class TavilySearchTests(unittest.TestCase):
    def test_normalizes_valid_results_and_preserves_provider_metadata(self) -> None:
        sources = normalize_tavily_results(
            {
                "results": [
                    {
                        "title": "RAG update",
                        "url": "https://example.com/rag",
                        "content": "A search snippet.",
                    }
                ]
            },
            max_results=3,
        )

        self.assertEqual(sources[0].label, "[W1]")
        self.assertEqual(sources[0].title, "RAG update")
        self.assertEqual(sources[0].url, "https://example.com/rag")
        self.assertEqual(sources[0].content, "A search snippet.")
        self.assertEqual(sources[0].rank, 1)

    def test_prefers_raw_content_and_caps_it_before_prompting(self) -> None:
        long_content = "a" * (MAX_CONTENT_CHARS_PER_SOURCE + 20)
        sources = normalize_tavily_results(
            {"results": [{"url": "https://example.com", "raw_content": long_content}]},
            max_results=1,
        )

        self.assertEqual(sources[0].title, "https://example.com")
        self.assertEqual(len(sources[0].content), MAX_CONTENT_CHARS_PER_SOURCE)

    def test_ignores_malformed_and_duplicate_results(self) -> None:
        sources = normalize_tavily_results(
            {
                "results": [
                    "not a result",
                    {"title": "Missing URL", "content": "text"},
                    {"url": "https://example.com", "content": "first"},
                    {"url": "https://example.com", "content": "duplicate"},
                    {"url": "https://other.example", "content": "second"},
                ]
            },
            max_results=3,
        )

        self.assertEqual([source.url for source in sources], ["https://example.com", "https://other.example"])
        self.assertEqual([source.rank for source in sources], [1, 2])

    def test_handles_unexpected_provider_response_shape(self) -> None:
        self.assertEqual(normalize_tavily_results({"results": "unexpected"}, 3), [])
        self.assertEqual(normalize_tavily_results("not JSON", 3), [])


if __name__ == "__main__":
    unittest.main()
