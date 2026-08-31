"""MediaWiki search and wikipedia-api fetch (article text used as-is)."""

from __future__ import annotations

import os

import requests
import wikipediaapi

# Wikimedia throttles generic/placeholder user agents harder, so allow a real
# contact address to be supplied via the environment.
USER_AGENT = os.getenv("NWP_USER_AGENT", "NextWordPredictor/1.0 (sam@gmail.com)")
MIN_TEXT_LENGTH = 200


class WikipediaFetchError(Exception):
    """Raised when a page cannot be used for training."""


class WikipediaSearchError(Exception):
    """Raised when the MediaWiki search request itself fails."""


def search_wikipedia(query: str, limit: int = 6) -> list[str]:
    """
    Uses Wikipedia's full-text search engine (action=query & list=search)
    to find matching articles for any keyword.

    Raises WikipediaSearchError when the request fails, so callers can tell a
    genuinely empty result set apart from a broken request.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "utf8": "",
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
    except requests.RequestException as exc:
        raise WikipediaSearchError(
            f"Could not reach the Wikipedia search API ({type(exc).__name__}): {exc}"
        ) from exc

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        wait_note = f" Retry after {retry_after}s." if retry_after else " Wait a minute and retry."
        raise WikipediaSearchError(
            "Wikipedia is rate limiting this client (HTTP 429)." + wait_note +
            " Set NWP_USER_AGENT to a user agent with a real contact address to reduce throttling. "
            f"Wikipedia said: {response.text[:300].strip()}"
        )

    if response.status_code >= 400:
        raise WikipediaSearchError(
            f"Wikipedia search API returned HTTP {response.status_code}: "
            f"{response.text[:300].strip()}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise WikipediaSearchError(
            f"Wikipedia search API returned a non-JSON response: {response.text[:300].strip()}"
        ) from exc

    if isinstance(data.get("error"), dict):
        error = data["error"]
        raise WikipediaSearchError(
            f"Wikipedia API error '{error.get('code', 'unknown')}': "
            f"{error.get('info', 'no details provided')}"
        )

    search_results = data.get("query", {}).get("search", [])
    return [item["title"] for item in search_results if item.get("title")]


def fetch_page(title: str) -> str:
    """
    Fetch a Wikipedia page by exact title.

    Returns page.text with no preprocessing. Requires the page to exist
    and have more than MIN_TEXT_LENGTH non-whitespace characters.
    """
    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")
    page = wiki.page(title)

    if not page.exists() or len(page.text.strip()) <= MIN_TEXT_LENGTH:
        raise WikipediaFetchError(
            f"'{title}' does not exist or does not have enough text for training."
        )

    return page.text
