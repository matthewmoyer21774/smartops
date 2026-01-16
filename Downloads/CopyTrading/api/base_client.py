"""
Rate-limited HTTP client for Polymarket APIs.
"""

import requests
import time
from typing import Optional, Dict, Any, List

import sys
sys.path.insert(0, '..')
from config import REQUESTS_PER_MINUTE, MAX_RETRIES, RETRY_DELAY


class RateLimitedClient:
    """HTTP client with rate limiting and retry logic."""

    def __init__(
        self,
        base_url: str,
        requests_per_minute: int = REQUESTS_PER_MINUTE,
        max_retries: int = MAX_RETRIES,
        retry_delay: float = RETRY_DELAY
    ):
        self.base_url = base_url.rstrip('/')
        self.min_interval = 60.0 / requests_per_minute
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CopyTrading/1.0',
            'Accept': 'application/json'
        })

    def _wait_for_rate_limit(self):
        """Ensure minimum interval between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> Optional[Any]:
        """
        Make GET request with retry logic.

        Returns parsed JSON or None on failure.
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            self._wait_for_rate_limit()

            try:
                response = self.session.get(url, params=params, timeout=timeout)
                self.last_request_time = time.time()

                if response.status_code == 429:  # Rate limited
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                if response.status_code == 404:
                    return None

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        return None

    def get_paginated(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        limit_key: str = 'limit',
        offset_key: str = 'offset',
        page_size: int = 500,
        max_items: Optional[int] = None
    ) -> List[Any]:
        """
        Fetch paginated results.

        Returns combined list of all results.
        """
        all_results = []
        offset = 0
        params = params or {}

        while True:
            params[limit_key] = page_size
            params[offset_key] = offset

            result = self.get(endpoint, params=params)

            if not result:
                break

            # Handle both list responses and dict with data key
            if isinstance(result, list):
                items = result
            elif isinstance(result, dict):
                items = result.get('data', result.get('results', []))
                if not isinstance(items, list):
                    items = [result]
            else:
                break

            all_results.extend(items)

            if len(items) < page_size:
                break

            if max_items and len(all_results) >= max_items:
                all_results = all_results[:max_items]
                break

            offset += page_size

        return all_results
