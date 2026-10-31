"""Shared fixtures for the DeepRetrieval unit tests.

Settings read from the environment (data directory, PubMed key) are cleared for every test, so
a developer's shell or a sourced `code/.env` cannot change what the suite observes.
"""
import pytest


@pytest.fixture(autouse=True)
def _hermetic_settings_env(monkeypatch):
    monkeypatch.delenv("DEEPRETRIEVAL_DATA_DIR", raising=False)
    monkeypatch.delenv("PUBMED_API_KEY", raising=False)
