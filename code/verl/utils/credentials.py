"""Secrets read from the environment, with the legacy key files as fallback.

The launchers source `code/.env` (written by the setup script), so on a pod the PubMed key
arrives as $PUBMED_API_KEY. Keys are never logged; only where one came from.
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PUBMED_API_KEY_ENV = "PUBMED_API_KEY"
# Legacy location (gitignored `*.key`), resolved from the package, not the cwd.
DEFAULT_PUBMED_API_KEY_PATH = Path(__file__).resolve().parent / "reward_score" / "apis" / "pubmed_api.key"


def load_pubmed_api_key(key_path: "str | os.PathLike[str] | None" = None) -> "str | None":
    """$PUBMED_API_KEY if non-blank, else the key file's stripped contents, else None (keyless).

    `key_path` defaults to `DEFAULT_PUBMED_API_KEY_PATH` (looked up at call time).
    """
    from_env = os.environ.get(PUBMED_API_KEY_ENV, "").strip()
    if from_env:
        logger.info("Using the PubMed API key from $%s", PUBMED_API_KEY_ENV)
        return from_env
    path = Path(key_path) if key_path is not None else DEFAULT_PUBMED_API_KEY_PATH
    if path.is_file():
        from_file = path.read_text().strip()
        if from_file:
            logger.info("Using the PubMed API key from %s", path)
            return from_file
        logger.warning("PubMed API key file %s is empty; ignoring it", path)
    return None
