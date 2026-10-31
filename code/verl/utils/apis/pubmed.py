import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_API_KEY_IN_TEXT = re.compile(r"(api_key=)[^&\s'\"]+")


def _redact(text: str) -> str:
    """Hide `api_key=...` values: requests puts the full request URL into its error messages."""
    return _API_KEY_IN_TEXT.sub(r"\1***", text)


class _RedactApiKeyFilter(logging.Filter):
    """Redact API keys from this module's log records, tracebacks included."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact(record.getMessage())
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = _redact(logging.Formatter().formatException(record.exc_info))
        return True


logger.addFilter(_RedactApiKeyFilter())

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="
PUBMED_SUMMARY_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id="
PUBMED_EFETCH_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="
REQUEST_TIMEOUT = (5, 60)  # (connect, read) seconds; a timeout raises and goes through the retry loop
# NCBI E-utilities allow 3 requests/s without an API key and 10/s with one (per IP / per key).
KEYLESS_REQUEST_DELAY = 1 / 3 + 0.01
KEYED_REQUEST_DELAY = 0.1 + 0.01


def _check_ncbi_payload(payload: dict, what: str) -> dict:
    """NCBI answers some failures (e.g. rate limiting) with HTTP 200 and {"error": ...}; raise on those."""
    if "error" in payload:
        raise ConnectionError(f"PubMed {what} returned an error: {payload['error']}")
    return payload


class PubmedAPI:
    """A wrapper class for the Pubmed API with optimized performance.
    """
    def __init__(self, retry=1, api_key=None, request_delay=None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        """`request_delay` is the minimum spacing between request starts; by default it follows
        NCBI's limit for the key (≥1/3 s keyless, ≥0.1 s keyed). Pass a smaller value only in
        tests or with a raised NCBI quota. `clock`/`sleep` are injectable for tests."""
        assert retry >= 0, f"retry must be >= 0, got {retry}"
        self.retry = retry
        self.api_key = api_key
        if request_delay is None:
            request_delay = KEYED_REQUEST_DELAY if api_key else KEYLESS_REQUEST_DELAY
        assert request_delay >= 0, f"request_delay must be >= 0, got {request_delay}"
        self.request_delay = request_delay
        self._clock = clock
        self._sleep = sleep
        self._throttle_lock = threading.Lock()  # reward code may share one client across threads
        self._last_request_at: float | None = None

    def _get(self, url):
        """`requests.get` spaced at least `request_delay` after the previous request started."""
        with self._throttle_lock:
            if self._last_request_at is not None:
                wait = self._last_request_at + self.request_delay - self._clock()
                if wait > 0:
                    self._sleep(wait)
            self._last_request_at = self._clock()
        return requests.get(url, timeout=REQUEST_TIMEOUT)

    def search(self, query, topk=-1):
        """Get the response of pmids."""
        search_results = self.search_with_query(query, topk=topk)
        return search_results

    def search_with_query(self, query, topk=-1):
        """Search with query input to get the response of pmids. Optimized for speed."""
        err_msg = ""
        pmid_list = []  # rebound by every attempt; retry >= 0 means at least one
        if self.api_key:
            query += f"&api_key={self.api_key}"
        if '&retmode=json' not in query:
            query += '&retmode=json'

        # Optimize retmax based on topk - no need to fetch large batches if topk is small
        # For unlimited retrieval or large topk values, use larger batch sizes
        if topk > 0 and topk <= 1000:
            retmax = topk  # If we need few results, just fetch them in one go
        else:
            retmax = 5000  # Higher value for better throughput, up to API limits
        
        original_query = query  # Save the original query without retmax/retstart parameters

        for i in range(self.retry + 1):
            if i > 0:
                logger.warning("Retry %d times", i)
                self._sleep(self.request_delay * 2)  # Longer delay on retries
            # Each attempt starts from scratch so a failed attempt leaves nothing behind
            retstart = 0
            pmid_list = []
            try:
                # First get the count of results
                count_query = original_query + '&retmax=0'
                response = self._get(count_query)
                
                if response.status_code != 200:
                    raise ConnectionError(f"Pubmed connection error occurred - {response.text}")
                
                response_dict = _check_ncbi_payload(json.loads(response.text), "search")
                total_count = int(response_dict['esearchresult']['count'])
                
                # If topk specified, limit to that number
                count_to_fetch = min(total_count, topk) if topk > 0 else total_count
                
                # No need to continue if no results or topk=0
                if count_to_fetch <= 0:
                    return []
                
                # Calculate how many batches we need
                batch_limit = min(10000, count_to_fetch)  # PubMed has a hard limit around 10,000
                
                # Fetch results in batches
                while retstart < batch_limit and len(pmid_list) < count_to_fetch:
                    # Adjust retmax for final batch if needed
                    current_retmax = min(retmax, count_to_fetch - len(pmid_list))
                    current_query = original_query + f'&retmax={current_retmax}&retstart={retstart}'
                    
                    response = self._get(current_query)
                    
                    if response.status_code != 200:
                        raise ConnectionError(f"Pubmed connection error occurred - {response.text}")
                    
                    response_dict = _check_ncbi_payload(json.loads(response.text), "search")
                    batch_results = response_dict['esearchresult']['idlist']
                    
                    # If no results returned in this batch, break the loop
                    if not batch_results:
                        break
                    
                    pmid_list.extend(batch_results)
                    retstart += current_retmax
                    
                    # If we have enough results, stop fetching
                    if topk > 0 and len(pmid_list) >= topk:
                        pmid_list = pmid_list[:topk]
                        break
                
                # We only need to remove duplicates when fetching unlimited results
                if topk <= 0:
                    pmid_list = list(dict.fromkeys(pmid_list))  # Remove duplicates while preserving order
                
                err_msg = ""  # this attempt succeeded; earlier failures no longer matter
                break
            except Exception as e:
                err_msg = _redact(f"{type(e).__name__}: {e}")
                logger.warning("PubMed API search attempt %d/%d failed: %s", i + 1, self.retry + 1, err_msg,
                               exc_info=True)

        if err_msg != "":
            raise RuntimeError(f"A Pubmed API error occurred: {err_msg}")

        return pmid_list
    
    def search_with_keywords(self, keywords: str, topk=-1):
        """Search with keywords input to get the response of pmids."""
        query = PUBMED_BASE_URL + keywords
        search_results = self.search_with_query(query, topk=topk)
        return search_results
    
    def get_papers_by_pmids(self, pmid_list):
        """Search pmids to get the summary of paper."""
        if not pmid_list:
            return pd.DataFrame()

        err_msg = ""
        papers = pd.DataFrame()  # rebound on success; on failure we raise below
        
        for i in range(self.retry + 1):
            if i > 0:
                logger.warning("Retry %d times", i)
                self._sleep(self.request_delay * 2)  # Longer delay on retries
            try:
                # Process in batches to avoid URL length limitations
                batch_size = 200  # Keep batch size reasonable to avoid HTTP 414 errors
                all_papers = []
                
                for i in range(0, len(pmid_list), batch_size):
                    batch_pmids = pmid_list[i:i+batch_size]
                    
                    # Build the query to get the summary of the articles
                    pmid_list_str = ','.join(batch_pmids)
                    summary_query = PUBMED_SUMMARY_BASE_URL + pmid_list_str + "&retmode=json"
                    if self.api_key:
                        summary_query += f"&api_key={self.api_key}"
                    
                    response = self._get(summary_query)

                    if response.status_code != 200:
                        if response.status_code == 414:
                            raise ConnectionError("Pubmed query too long!")
                        else:
                            raise ConnectionError(f"Pubmed connection error occurred: {response.text}")

                    response = _check_ncbi_payload(json.loads(response.text), "summary")
                    results = response.get("result", {})
                    uids = results.get("uids", [])

                    if len(uids) == 0:
                        continue
                    
                    # Build dataframe for this batch
                    batch_papers = []
                    for idx in range(len(uids)):
                        try:
                            cur_res = results[uids[idx]]
                            parse_res = self._parse_json_summary_response(cur_res)
                            batch_papers.append(parse_res)
                        except Exception:
                            logger.warning("Error parsing paper %s", uids[idx], exc_info=True)
                    
                    if batch_papers:
                        batch_df = pd.concat(batch_papers, axis=0).reset_index(drop=True)
                        
                        # Retrieve abstract and mesh terms for this batch
                        abstracts, mesh_terms = self._retrieve_abstract_and_mesh_term_by_efetch(uids)
                        batch_df['Abstract'] = abstracts
                        batch_df['Mesh Term'] = mesh_terms
                        
                        all_papers.append(batch_df)
                
                # Combine all batches
                if all_papers:
                    papers = pd.concat(all_papers, axis=0).reset_index(drop=True)
                else:
                    papers = pd.DataFrame()
                
                err_msg = ""  # this attempt succeeded; earlier failures no longer matter
                break
            except Exception as e:
                err_msg = _redact(f"{type(e).__name__}: {e}")
                logger.warning("PubMed get_papers_by_pmids attempt %d/%d failed: %s", i + 1, self.retry + 1,
                               err_msg, exc_info=True)
        
        if err_msg != "":
            raise RuntimeError(f"A Pubmed API error occurred: {err_msg}")

        return papers
    
    def _parse_json_summary_response(self, res):
        """Parse the summary json response to pd dataframe from the API call."""
        pmid = res.get("uid", None)
        pub_date = res.get("pubdate", None)
        journal = res.get("fulljournalname", None)
        if journal is None:
            journal = res.get("source", None)
        title = res.get("title", None)
        authors = res.get("authors", [])
        authors = [author.get("name", None) for author in authors]
        authors = "; ".join(authors)
        volume = res.get("volume", None)
        issue = res.get("issue", None)
        pages = res.get("pages", None)
        pubtypes = res.get("pubtype", [])
        url = 'https://pubmed.ncbi.nlm.nih.gov/' + pmid

        df = pd.DataFrame({
            "PMID": [pmid],
            "Publication Date": [pub_date],
            "Title": [title],
            "Authors": [authors],
            "Journal": [journal],
            "Volume": [volume],
            "Issue": [issue],
            "Pages": [pages],
            "Pubtypes": [pubtypes],
            "URL": [url]
        })
        return df
    
    def _retrieve_abstract_and_mesh_term_by_efetch(self, pmid_list):
        '''Retrieve abstract based on pmids from efetch API'''
        # Process in batches to avoid URL length limitations
        batch_size = 200
        all_abstracts = {}
        all_mesh_terms = {}
        
        for i in range(0, len(pmid_list), batch_size):
            batch_pmids = pmid_list[i:i+batch_size]
            pmid_list_str = ','.join(batch_pmids)
            
            query = PUBMED_EFETCH_BASE_URL + pmid_list_str + "&retmode=xml"
            if self.api_key:
                query += "&api_key=" + self.api_key

            response = self._get(query)
            
            if response.status_code != 200:
                logger.warning("Error fetching abstracts for batch %d-%d: HTTP %s", i, i + batch_size, response.status_code)
                continue
            
            try:
                response_text = response.text
                tree = ET.ElementTree(ET.fromstring(response_text))
                articles = tree.findall(".//PubmedArticle")
                
                for article in articles:
                    abstract = article.find(".//AbstractText")
                    if abstract is not None:
                        abstract_text = abstract.text
                    else:
                        abstract_text = ""

                    # Get pmid
                    article_ids = [a for a in article.findall(".//ArticleId") if a.get("IdType", "").lower() == "pubmed"]
                    if len(article_ids) > 0:
                        pmid = article_ids[0].text
                        all_abstracts[pmid] = abstract_text
                    else:
                        continue
                
                    # Get MeSH terms and Qualifiers
                    mesh_terms = []
                    mesh_headings = article.findall(".//MeshHeading")
                    for mesh_heading in mesh_headings:
                        descriptor = mesh_heading.find("DescriptorName")
                        if descriptor is not None:
                            qualifiers = mesh_heading.findall("QualifierName")
                            if qualifiers:
                                for qualifier in qualifiers:
                                    mesh_terms.append(f"{descriptor.text} ({qualifier.text})")
                            else:
                                mesh_terms.append(descriptor.text)

                    all_mesh_terms[pmid] = mesh_terms
            except Exception:
                logger.warning("Error parsing XML for batch %d-%d", i, i + batch_size, exc_info=True)
        
        # Make sure the input and output are in the same order and length
        output_abstracts = []
        output_mesh_terms = []
        for pmid in pmid_list:
            output_abstracts.append(all_abstracts.get(pmid, ""))
            output_mesh_terms.append(all_mesh_terms.get(pmid, []))

        return output_abstracts, output_mesh_terms