"""src/utils/{gpt,gpt_azure,claude_api,claude_aws}.py: credentials are read lazily, env first."""
import sys

import pytest

from tests.deepretrieval.evals._loader import load_script

GPT_STUBS = ("openai",)
AZURE_STUBS = (
    "openai",
    "langchain_openai",
    "langchain_community",
    "langchain_community.callbacks",
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.prompts",
    "langchain_core.output_parsers",
)
BEDROCK_STUBS = ("boto3",)
ANTHROPIC_STUBS = ("anthropic", "langchain_anthropic", "langchain_core", "langchain_core.prompts")

AWS_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
# gpt_azure.py writes these into os.environ; registering them here makes monkeypatch undo that.
AZURE_ENV = ("AZURE_OPENAI_API_KEY", "OPENAI_API_TYPE", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION")


@pytest.fixture
def empty_cwd(tmp_path, monkeypatch):
    """A working directory with no key files and no credential env vars."""
    monkeypatch.chdir(tmp_path)
    for var in ("OPENAI_API_KEY", *AZURE_ENV, *AWS_ENV):
        # setenv first so the variable is restored (or removed) at teardown even if absent now.
        monkeypatch.setenv(var, "unset")
        monkeypatch.delenv(var)
    return tmp_path


@pytest.mark.parametrize(
    ("relpath", "stubs"),
    [
        ("src/utils/gpt.py", GPT_STUBS),
        ("src/utils/gpt_azure.py", AZURE_STUBS),
        ("src/utils/claude_api.py", BEDROCK_STUBS),
        ("src/utils/claude_aws.py", ANTHROPIC_STUBS),
    ],
)
def test_client_module_imports_without_key_files(monkeypatch, empty_cwd, relpath, stubs):
    module = load_script(monkeypatch, relpath, stubs)

    assert module.__file__ is not None
    assert module.__file__.endswith(relpath)


def test_gpt_chat_reuses_one_client_built_from_env_key(monkeypatch, empty_cwd):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    gpt = load_script(monkeypatch, "src/utils/gpt.py", GPT_STUBS)
    openai_cls = sys.modules["openai"].OpenAI

    gpt.gpt_chat("gpt-4o", "hi")
    gpt.gpt_chat_35("hi")

    openai_cls.assert_called_once_with(api_key="sk-env")


def test_gpt_chat_falls_back_to_key_file(monkeypatch, empty_cwd):
    (empty_cwd / "openai_api.key").write_text("sk-file\n")
    gpt = load_script(monkeypatch, "src/utils/gpt.py", GPT_STUBS)
    openai_cls = sys.modules["openai"].OpenAI

    gpt.gpt_chat("gpt-4o", "hi")

    openai_cls.assert_called_once_with(api_key="sk-file")


def test_gpt_chat_without_any_credential_names_both_sources(monkeypatch, empty_cwd):
    gpt = load_script(monkeypatch, "src/utils/gpt.py", GPT_STUBS)

    with pytest.raises(RuntimeError, match=r"OPENAI_API_KEY.*openai_api\.key"):
        gpt.gpt_chat("gpt-4o", "hi")


def test_gpt_azure_exports_key_from_file_on_first_call(monkeypatch, empty_cwd):
    (empty_cwd / "openai_api_azure.key").write_text("az-file\n")
    azure = load_script(monkeypatch, "src/utils/gpt_azure.py", AZURE_STUBS)

    azure.gpt_chat_4o("hi")

    assert azure.os.environ["AZURE_OPENAI_API_KEY"] == "az-file"


def test_gpt_azure_without_any_credential_names_both_sources(monkeypatch, empty_cwd):
    azure = load_script(monkeypatch, "src/utils/gpt_azure.py", AZURE_STUBS)

    with pytest.raises(RuntimeError, match=r"AZURE_OPENAI_API_KEY.*openai_api_azure\.key"):
        azure.gpt_chat_4o("hi")


def test_claude_api_prefers_aws_env_credentials(monkeypatch, empty_cwd):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-env")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-env")
    claude = load_script(monkeypatch, "src/utils/claude_api.py", BEDROCK_STUBS)
    client_factory = sys.modules["boto3"].client
    client_factory.return_value.invoke_model.return_value = {
        "body": _Body(b'{"content": [{"text": "ok"}]}')
    }

    text = claude.get_claude_response("haiku", "hi")
    claude.get_claude_response("haiku", "hi")

    assert text == "ok"
    client_factory.assert_called_once_with(
        "bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id="AKIA-env",
        aws_secret_access_key="secret-env",
    )


def test_claude_api_falls_back_to_key_files(monkeypatch, empty_cwd):
    (empty_cwd / "aws_access.key").write_text("AKIA-file\n")
    (empty_cwd / "claude_api_aws.key").write_text("secret-file\n")
    claude = load_script(monkeypatch, "src/utils/claude_api.py", BEDROCK_STUBS)
    client_factory = sys.modules["boto3"].client
    client_factory.return_value.invoke_model.return_value = {
        "body": _Body(b'{"content": [{"text": "ok"}]}')
    }

    claude.get_claude_response("sonnet", "hi")

    assert client_factory.call_args.kwargs["aws_access_key_id"] == "AKIA-file"


def test_claude_api_without_any_credential_names_both_sources(monkeypatch, empty_cwd):
    claude = load_script(monkeypatch, "src/utils/claude_api.py", BEDROCK_STUBS)

    with pytest.raises(RuntimeError, match=r"AWS_ACCESS_KEY_ID.*aws_access\.key"):
        claude.get_claude_response("haiku", "hi")


def test_claude_aws_reads_two_line_key_file_on_first_call(monkeypatch, empty_cwd):
    (empty_cwd / "claude_api_aws.key").write_text("AKIA-file\nsecret-file\n")
    claude = load_script(monkeypatch, "src/utils/claude_aws.py", ANTHROPIC_STUBS)
    bedrock_cls = sys.modules["anthropic"].AnthropicBedrock

    claude.chat_haiku("hi")
    claude.chat_sonnet("hi")

    bedrock_cls.assert_called_once_with(
        aws_access_key="AKIA-file", aws_secret_key="secret-file", aws_region="us-west-2"
    )


def test_claude_aws_without_any_credential_names_both_sources(monkeypatch, empty_cwd):
    claude = load_script(monkeypatch, "src/utils/claude_aws.py", ANTHROPIC_STUBS)

    with pytest.raises(RuntimeError, match=r"AWS_ACCESS_KEY_ID.*claude_api_aws\.key"):
        claude.chat_haiku("hi")


class _Body:
    """Stands in for botocore's StreamingBody: one read() of a fixed payload."""

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload
