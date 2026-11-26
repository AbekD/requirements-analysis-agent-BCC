import pytest
from google.genai import errors

from src.analysis.llm_client import GeminiClient, LLMError, LLMOutputError
from src.models.schemas import AssessmentLLM, Criterion, IndividualResponse, Status

GOOD = IndividualResponse(assessments=[AssessmentLLM(criterion=c, status=Status.PASS, explanation="ок")
                                       for c in Criterion]).model_dump_json()


class _Resp:
    def __init__(self, text):
        self.text = text


class FakeSDK:
    def __init__(self, script):
        self.script = list(script)
        self.prompts = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item)


def _client(sdk, sleeps):
    return GeminiClient("k", "m", requests_per_minute=600, sdk_client=sdk, sleep=sleeps.append, backoff_base=1.0)


def test_retries_invalid_json_then_succeeds_with_feedback():
    sdk, sleeps = FakeSDK(["не json", GOOD]), []
    result = _client(sdk, sleeps).generate(system="s", prompt="p", schema=IndividualResponse)
    assert len(result.assessments) == 10 and len(sdk.prompts) == 2
    assert "не прошёл проверку схемы" in sdk.prompts[1]


def test_gives_up_after_limited_json_retries():
    sdk = FakeSDK(["{}", "{}", "{}"])
    with pytest.raises(LLMOutputError):
        _client(sdk, []).generate(system="s", prompt="p", schema=IndividualResponse)
    assert len(sdk.prompts) == 3            # 1 + 2 повтора, не бесконечно


def test_empty_response_counts_as_invalid():
    sdk = FakeSDK([None, GOOD])
    assert _client(sdk, []).generate(system="s", prompt="p", schema=IndividualResponse)


def test_rate_limit_backoff_then_success():
    sdk, sleeps = FakeSDK([errors.ClientError(429, {"error": {"message": "quota"}}), GOOD]), []
    client = _client(sdk, sleeps)
    assert client.generate(system="s", prompt="p", schema=IndividualResponse)
    assert any(s >= 1.0 for s in sleeps) and client.calls == 2


def test_non_retryable_api_error_fails_fast():
    sdk = FakeSDK([errors.ClientError(400, {"error": {"message": "bad"}})])
    with pytest.raises(LLMError):
        _client(sdk, []).generate(system="s", prompt="p", schema=IndividualResponse)
    assert len(sdk.prompts) == 1


def test_server_errors_exhaust_retries():
    sdk = FakeSDK([errors.ServerError(503, {"error": {"message": "down"}})] * 5)
    with pytest.raises(LLMError):
        _client(sdk, []).generate(system="s", prompt="p", schema=IndividualResponse)
