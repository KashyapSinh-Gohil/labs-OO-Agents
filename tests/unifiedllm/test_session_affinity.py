# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""prompt_cache_key doubles as the provider-side session-affinity hint."""

import pytest
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import Choices, Message, ModelResponse

from nooa.unifiedllm import CompletionClient, ResponsesClient
from nooa.unifiedllm.cache_policy import SESSION_AFFINITY_HEADER, add_session_affinity_header


def test_header_mirrors_prompt_cache_key():
    params = {"prompt_cache_key": "agent-1-CodeActV2"}
    add_session_affinity_header(params)
    assert params["extra_headers"] == {SESSION_AFFINITY_HEADER: "agent-1-CodeActV2"}


@pytest.mark.parametrize("params", [{}, {"prompt_cache_key": ""}, {"prompt_cache_key": None}])
def test_no_key_adds_nothing(params):
    before = dict(params)
    add_session_affinity_header(params)
    assert params == before


def test_existing_headers_are_kept_and_an_explicit_value_wins():
    params = {
        "prompt_cache_key": "k",
        "extra_headers": {"x-other": "1", SESSION_AFFINITY_HEADER: "pinned-elsewhere"},
    }
    add_session_affinity_header(params)
    assert params["extra_headers"] == {"x-other": "1", SESSION_AFFINITY_HEADER: "pinned-elsewhere"}


def test_non_mapping_extra_headers_is_rejected():
    with pytest.raises(ValueError, match="extra_headers"):
        add_session_affinity_header({"prompt_cache_key": "k", "extra_headers": "nope"})


def _chat_response() -> ModelResponse:
    return ModelResponse(
        model="m",
        choices=[Choices(finish_reason="stop", message=Message(content="ok", role="assistant"))],
    )


def _responses_response() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        model="m",
        object="response",
        status="completed",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [None, "session-7-CodeActV2"])
async def test_completion_client_sends_the_header_only_with_a_key(monkeypatch, key):
    sent = {}

    async def fake(**kwargs):
        sent.update(kwargs)
        return _chat_response()

    monkeypatch.setattr("litellm.acompletion", fake)
    async with CompletionClient("openai/nvidia/moonshotai/kimi-k3", api_key="test") as client:
        extra = {"prompt_cache_key": key} if key else {}
        await client.acall([{"role": "user", "content": "hi"}], **extra)
    if key:
        assert sent["extra_headers"][SESSION_AFFINITY_HEADER] == key
    else:
        assert "extra_headers" not in sent


@pytest.mark.asyncio
async def test_responses_client_sends_the_header(monkeypatch):
    sent = {}

    async def fake(**kwargs):
        sent.update(kwargs)
        return _responses_response()

    monkeypatch.setattr("litellm.aresponses", fake)
    async with ResponsesClient("openai/nvidia/moonshotai/kimi-k3", api_key="test") as client:
        await client.acall(
            [{"role": "user", "content": "hi"}],
            prompt_cache_key="s-9",
            extra_headers={"x-other": "1"},
        )
    assert sent["extra_headers"] == {"x-other": "1", SESSION_AFFINITY_HEADER: "s-9"}
