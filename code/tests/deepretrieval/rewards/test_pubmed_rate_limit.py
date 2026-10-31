"""S3: PubmedAPI respects NCBI's rate limits and reports NCBI error payloads readably.

NCBI E-utilities allow 3 requests/s without an API key and 10/s with one. `requests.get` is
replaced (autospec) and the client gets an injected clock and sleep, so nothing sleeps for real.
"""
import json
import logging
import re
from itertools import pairwise
from unittest import mock

import pytest

from verl.utils.apis import pubmed

RATE_LIMIT_ERROR = {"error": "API rate limit exceeded", "api-key": "1.2.3.4", "count": "4", "limit": "3"}


class FakeClock:
    """A monotonic clock that only moves when the client sleeps."""

    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        assert 0 <= seconds < 60, f"implausible sleep {seconds}"
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.text = json.dumps(payload)


def _int_param(url, name, default: "int | None" = None) -> int:
    match = re.search(rf"[?&]{name}=(\d+)", url)
    if match:
        return int(match.group(1))
    assert default is not None, f"request has no {name}: {url}"
    return default


def _esearch_payload(url, total=5):
    retmax = _int_param(url, "retmax")
    start = _int_param(url, "retstart", default=0)
    ids = [str(i) for i in range(start, min(start + retmax, total))]
    return {"esearchresult": {"count": str(total), "idlist": ids}}


def _client(clock, **kwargs):
    return pubmed.PubmedAPI(retry=1, clock=clock.time, sleep=clock.sleep, **kwargs)


def _patch_get(fake):
    return mock.patch.object(pubmed.requests, "get", autospec=True, side_effect=fake)


def _request_times(clock):
    times = []

    def get(url, **kwargs):
        times.append(clock.now)
        return FakeResponse(_esearch_payload(url))

    return times, get


def test_keyless_client_waits_at_least_a_third_of_a_second_between_requests():
    assert pubmed.PubmedAPI().request_delay >= 1 / 3


def test_keyed_client_waits_at_least_a_tenth_of_a_second_between_requests():
    delay = pubmed.PubmedAPI(api_key="k").request_delay

    assert 0.1 <= delay < 1 / 3


def test_explicit_request_delay_is_honoured():
    assert pubmed.PubmedAPI(request_delay=0.5).request_delay == pytest.approx(0.5)


@pytest.mark.parametrize(("api_key", "min_gap"), [(None, 1 / 3), ("k", 0.1)])
def test_consecutive_requests_are_spaced_by_the_rate_limit(api_key, min_gap):
    clock = FakeClock()
    api = _client(clock, api_key=api_key)
    times, get = _request_times(clock)

    with _patch_get(get):
        api.search_with_keywords("cancer", topk=3000)  # count request + one batch request
        api.search_with_keywords("asthma", topk=3000)

    gaps = [later - earlier for earlier, later in pairwise(times)]
    assert len(times) == 4
    assert min(gaps) >= min_gap - 1e-9


def test_first_request_does_not_wait():
    clock = FakeClock()
    api = _client(clock)
    times, get = _request_times(clock)

    with _patch_get(get):
        api.search_with_keywords("cancer", topk=3000)

    assert times[0] == pytest.approx(100.0)  # the fake clock never advanced


def test_rate_limit_error_payload_is_retried(caplog):
    clock = FakeClock()
    api = _client(clock)
    replies = iter([FakeResponse(RATE_LIMIT_ERROR)])

    def get(url, **kwargs):
        return next(replies, None) or FakeResponse(_esearch_payload(url))

    with _patch_get(get), caplog.at_level(logging.WARNING, logger=pubmed.__name__):
        pmids = api.search_with_keywords("cancer", topk=3000)

    assert pmids == ["0", "1", "2", "3", "4"]
    assert "API rate limit exceeded" in caplog.text
    assert "KeyError" not in caplog.text


def test_persistent_error_payload_raises_with_the_ncbi_message():
    clock = FakeClock()
    api = _client(clock)

    with _patch_get(lambda url, **kwargs: FakeResponse(RATE_LIMIT_ERROR)), \
            pytest.raises(RuntimeError, match="API rate limit exceeded"):
        api.search_with_keywords("cancer", topk=3000)


def test_error_payload_in_a_batch_response_is_reported():
    clock = FakeClock()
    api = pubmed.PubmedAPI(retry=0, clock=clock.time, sleep=clock.sleep)
    replies = iter([FakeResponse({"esearchresult": {"count": "5", "idlist": []}}), FakeResponse(RATE_LIMIT_ERROR)])

    with _patch_get(lambda url, **kwargs: next(replies)), pytest.raises(RuntimeError, match="API rate limit exceeded"):
        api.search_with_keywords("cancer", topk=3000)


def test_error_payload_in_a_summary_response_is_reported():
    clock = FakeClock()
    api = pubmed.PubmedAPI(retry=0, clock=clock.time, sleep=clock.sleep)

    with _patch_get(lambda url, **kwargs: FakeResponse(RATE_LIMIT_ERROR)), \
            pytest.raises(RuntimeError, match="API rate limit exceeded"):
        api.get_papers_by_pmids(["7"])


def test_retry_backoff_uses_the_injected_sleep():
    clock = FakeClock()
    api = _client(clock)

    with _patch_get(lambda url, **kwargs: FakeResponse(RATE_LIMIT_ERROR)), pytest.raises(RuntimeError):
        api.search_with_keywords("cancer", topk=3000)

    assert clock.sleeps, "the client waited without the injected sleep"
    assert sum(clock.sleeps) >= 2 * (1 / 3)  # retry backoff is twice the request spacing


def test_api_key_never_appears_in_logs(caplog):
    clock = FakeClock()
    api = _client(clock, api_key="very-secret-key")

    with _patch_get(lambda url, **kwargs: FakeResponse(RATE_LIMIT_ERROR)), \
            caplog.at_level(logging.DEBUG, logger=pubmed.__name__), pytest.raises(RuntimeError) as excinfo:
        api.search_with_keywords("cancer", topk=3000)

    assert "very-secret-key" not in caplog.text
    assert "very-secret-key" not in str(excinfo.value)


def test_api_key_in_a_transport_error_is_redacted_from_logs_and_the_raised_error(caplog):
    clock = FakeClock()
    api = _client(clock, api_key="very-secret-key")

    def get(url, **kwargs):  # requests puts the request URL (with &api_key=...) in its error text
        raise pubmed.requests.ConnectionError(f"Max retries exceeded with url: {url}")

    with _patch_get(get), caplog.at_level(logging.DEBUG, logger=pubmed.__name__), \
            pytest.raises(RuntimeError) as excinfo:
        api.search_with_keywords("cancer", topk=3000)

    assert "Max retries exceeded" in caplog.text  # the failure is still reported...
    assert "very-secret-key" not in caplog.text  # ...without the key
    assert "very-secret-key" not in str(excinfo.value)
