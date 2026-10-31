"""A4, B15, B16: retry, timeout and paging behaviour of the PubMed / CT.gov HTTP clients.

`requests.get` is replaced (autospec) by small fake servers; nothing touches the network.
"""
import json
import logging
import re
from unittest import mock

import pytest
import requests

from verl.utils.apis import ctgov, pubmed


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.text = payload if isinstance(payload, str) else json.dumps(payload)


def _param(url, name):
    match = re.search(rf"[?&]{name}=([^&]*)", url)
    return match.group(1) if match else None


def _required_param(url, name):
    value = _param(url, name)
    assert value is not None, f"request has no {name} parameter: {url}"
    return value


class FakeESearch:
    """PubMed esearch: `total` ids; listed calls (0-based) raise `failure` instead of answering."""

    def __init__(self, total, failing_calls=(), failure=None):
        self.total = total
        self.failing_calls = set(failing_calls)
        self.failure = failure or ConnectionResetError("reset")
        self.calls = 0

    def __call__(self, url, **kwargs):
        call, self.calls = self.calls, self.calls + 1
        if call in self.failing_calls:
            raise self.failure
        retmax = int(_required_param(url, "retmax"))
        retstart = int(_param(url, "retstart") or 0)
        ids = [str(i) for i in range(retstart, min(retstart + retmax, self.total))]
        return FakeResponse({"esearchresult": {"count": str(self.total), "idlist": ids}})


class FakeCTGov:
    """CT.gov v2 studies endpoint with `total` studies, paged by pageSize; listed calls raise."""

    def __init__(self, total, failing_calls=(), endless=False, call_budget=100, failure=None):
        self.total = total
        self.failing_calls = set(failing_calls)
        self.failure = failure or ConnectionResetError("reset")
        self.endless = endless
        self.call_budget = call_budget
        self.calls = 0

    def __call__(self, url, **kwargs):
        call, self.calls = self.calls, self.calls + 1
        if call >= self.call_budget:
            raise AssertionError("fake CT.gov call budget exhausted: client paged without a bound")
        if call in self.failing_calls:
            raise self.failure
        page_size = int(_required_param(url, "pageSize"))
        start = int(_param(url, "pageToken") or 0)
        end = start + page_size if self.endless else min(start + page_size, self.total)
        studies = [{"protocolSection": {"identificationModule": {"nctId": f"NCT{i:08d}"}}}
                   for i in range(start, end)]
        next_token = str(end) if (self.endless or end < self.total) else ""
        return FakeResponse({"totalCount": self.total, "studies": studies, "nextPageToken": next_token})


def _patch_get(module, fake):
    return mock.patch.object(module.requests, "get", autospec=True, side_effect=fake)


def test_pubmed_search_returns_results_when_a_retry_succeeds():
    api = pubmed.PubmedAPI(retry=1, request_delay=0)
    server = FakeESearch(total=6000, failing_calls={2})  # count, batch 1, then batch 2 fails

    with _patch_get(pubmed, server):
        pmids = api.search_with_keywords("cancer", topk=-1)

    assert pmids == [str(i) for i in range(6000)]


def test_pubmed_search_raises_when_every_attempt_fails():
    api = pubmed.PubmedAPI(retry=1, request_delay=0)
    server = FakeESearch(total=10, failing_calls={0, 1})

    with _patch_get(pubmed, server), pytest.raises(RuntimeError, match="Pubmed API error"):
        api.search_with_keywords("cancer", topk=3000)


def test_pubmed_search_retries_after_a_timeout():
    api = pubmed.PubmedAPI(retry=1, request_delay=0)
    server = FakeESearch(total=5, failing_calls={0}, failure=requests.Timeout("slow"))  # audit: ignore[NETWORK] builds the exception only, no request

    with _patch_get(pubmed, server):
        pmids = api.search_with_keywords("cancer", topk=3000)

    assert pmids == ["0", "1", "2", "3", "4"]


@pytest.mark.parametrize("module", [pubmed, ctgov])
def test_request_timeout_is_five_second_connect_sixty_second_read(module):
    assert module.REQUEST_TIMEOUT == (5, 60)


def test_pubmed_search_passes_a_timeout_to_every_request():
    api = pubmed.PubmedAPI(retry=1, request_delay=0)

    with _patch_get(pubmed, FakeESearch(total=5)) as get:
        api.search_with_keywords("cancer", topk=3000)

    assert [c.kwargs.get("timeout") for c in get.call_args_list] == [pubmed.REQUEST_TIMEOUT] * 2


SUMMARY = {"result": {"uids": ["7"], "7": {"uid": "7", "title": "T", "authors": [{"name": "A"}]}}}
EFETCH = ("<PubmedArticleSet><PubmedArticle><AbstractText>abs</AbstractText>"
          "<ArticleId IdType=\"pubmed\">7</ArticleId></PubmedArticle></PubmedArticleSet>")


def _fake_paper_server(fail_first_summary):
    state = {"summary_calls": 0}

    def get(url, **kwargs):
        if "esummary" in url:
            state["summary_calls"] += 1
            if fail_first_summary and state["summary_calls"] == 1:
                return FakeResponse("busy", status_code=503)
            return FakeResponse(SUMMARY)
        return FakeResponse(EFETCH)

    return get


def test_pubmed_get_papers_returns_papers_when_a_retry_succeeds():
    api = pubmed.PubmedAPI(retry=1, request_delay=0)

    with _patch_get(pubmed, _fake_paper_server(fail_first_summary=True)):
        papers = api.get_papers_by_pmids(["7"])

    assert papers["Abstract"].tolist() == ["abs"]


def test_pubmed_get_papers_passes_a_timeout_to_every_request():
    api = pubmed.PubmedAPI(retry=1, request_delay=0)

    with _patch_get(pubmed, _fake_paper_server(fail_first_summary=False)) as get:
        api.get_papers_by_pmids(["7"])

    assert [c.kwargs.get("timeout") for c in get.call_args_list] == [pubmed.REQUEST_TIMEOUT] * 2


def test_ctgov_search_does_not_duplicate_ids_when_a_retry_succeeds():
    api = ctgov.CTGovAPI(retry=1, request_delay=0)
    server = FakeCTGov(total=2500, failing_calls={1})  # page 1 ok, page 2 fails, retry from scratch

    with _patch_get(ctgov, server):
        nctids = api.search_with_keywords("asthma", topk=-1)

    assert nctids == [f"NCT{i:08d}" for i in range(2500)]


def test_ctgov_search_raises_instead_of_returning_partial_results_when_every_attempt_fails():
    api = ctgov.CTGovAPI(retry=1, request_delay=0)
    server = FakeCTGov(total=2500, failing_calls={1, 3})  # page 2 fails on both attempts

    with _patch_get(ctgov, server), pytest.raises(RuntimeError, match="CTGOV API error"):
        api.search_with_keywords("asthma", topk=-1)


def test_ctgov_search_topk_3000_returns_exactly_3000_ids_in_three_pages():
    api = ctgov.CTGovAPI(retry=1, request_delay=0)
    server = FakeCTGov(total=5000)

    with _patch_get(ctgov, server) as get:
        nctids = api.search_with_keywords("asthma", topk=3000)

    assert nctids == [f"NCT{i:08d}" for i in range(3000)]
    assert get.call_count == 3


def test_ctgov_search_raises_when_paging_exceeds_max_pages(monkeypatch):
    monkeypatch.setattr(ctgov, "MAX_PAGES", 3, raising=False)
    api = ctgov.CTGovAPI(retry=0, request_delay=0)

    with _patch_get(ctgov, FakeCTGov(total=0, endless=True)), \
            pytest.raises(RuntimeError, match=r"more than 3 pages"):
        api.search_with_keywords("asthma", topk=-1)


def test_ctgov_get_trials_raises_when_paging_exceeds_max_pages(monkeypatch):
    monkeypatch.setattr(ctgov, "MAX_PAGES", 3, raising=False)
    api = ctgov.CTGovAPI(retry=0, request_delay=0)

    with _patch_get(ctgov, FakeCTGov(total=0, endless=True)), \
            pytest.raises(RuntimeError, match=r"more than 3 pages"):
        api.get_trials_by_nctids(["NCT00000001"])


def test_ctgov_search_passes_a_timeout_to_every_request():
    api = ctgov.CTGovAPI(retry=1, request_delay=0)

    with _patch_get(ctgov, FakeCTGov(total=1500)) as get:
        api.search_with_keywords("asthma", topk=-1)

    assert [c.kwargs.get("timeout") for c in get.call_args_list] == [ctgov.REQUEST_TIMEOUT] * 2


def test_ctgov_get_trials_passes_a_timeout_to_every_request():
    api = ctgov.CTGovAPI(retry=1, request_delay=0)

    with _patch_get(ctgov, FakeCTGov(total=1)) as get:
        api.get_trials_by_nctids(["NCT00000000"])

    assert [c.kwargs.get("timeout") for c in get.call_args_list] == [ctgov.REQUEST_TIMEOUT]


def test_ctgov_search_retries_after_a_timeout():
    api = ctgov.CTGovAPI(retry=1, request_delay=0)
    server = FakeCTGov(total=2, failing_calls={0}, failure=requests.Timeout("slow"))  # audit: ignore[NETWORK] builds the exception only, no request

    with _patch_get(ctgov, server):
        nctids = api.search_with_keywords("asthma", topk=3000)

    assert nctids == ["NCT00000000", "NCT00000001"]


# ---------------------------------------------------------------- logging (audit 3, item 6)


def _warnings_from(caplog, module):
    return [r for r in caplog.records if r.name == module.__name__ and r.levelno == logging.WARNING]


def test_pubmed_search_logs_retry_at_warning(caplog):
    api = pubmed.PubmedAPI(retry=1, request_delay=0)

    with _patch_get(pubmed, FakeESearch(total=5, failing_calls={0})), caplog.at_level(logging.DEBUG):
        api.search_with_keywords("cancer", topk=3000)

    assert any("Retry 1" in r.getMessage() for r in _warnings_from(caplog, pubmed))


def test_pubmed_search_logs_failed_attempt_at_warning_with_traceback(caplog):
    api = pubmed.PubmedAPI(retry=1, request_delay=0)

    with _patch_get(pubmed, FakeESearch(total=5, failing_calls={0})), caplog.at_level(logging.DEBUG):
        api.search_with_keywords("cancer", topk=3000)

    assert [r.exc_info[0] for r in _warnings_from(caplog, pubmed) if r.exc_info] == [ConnectionResetError]


def test_ctgov_search_logs_retry_at_warning(caplog):
    api = ctgov.CTGovAPI(retry=1, request_delay=0)

    with _patch_get(ctgov, FakeCTGov(total=10, failing_calls={0})), caplog.at_level(logging.DEBUG):
        api.search_with_keywords("asthma", topk=-1)

    assert any("Retry 1" in r.getMessage() for r in _warnings_from(caplog, ctgov))


def test_api_retry_writes_nothing_to_stdout(capsys):
    api = pubmed.PubmedAPI(retry=1, request_delay=0)

    with _patch_get(pubmed, FakeESearch(total=5, failing_calls={0})):
        api.search_with_keywords("cancer", topk=3000)

    assert capsys.readouterr().out == ""
