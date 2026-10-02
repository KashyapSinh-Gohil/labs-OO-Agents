# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit tool choice survives the SDK/HTTP boundary with cache schemas intact."""

import copy
import json

import httpx
import pytest

from nooa.unifiedllm import CacheBoundary, CompletionClient, ResponsesClient, RetryConfig, Tool
from nooa.unifiedllm.unifiedllm import _ClientHttp


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["responses", "anthropic"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("configured", [False, True], ids=["per-call", "constructor"])
@pytest.mark.parametrize("choice", [None, "auto", "none", "required", "named"])
async def test_explicit_choice_on_wire(monkeypatch, style, is_async, configured, choice):
    bodies = []
    if choice == "named":
        choice = (
            {"type": "function", "name": "lookup"}
            if style == "responses"
            else {"type": "function", "function": {"name": "lookup"}}
        )

    def respond(request):
        bodies.append(json.loads(request.content))
        if style == "responses":
            assert request.url.path == "/v1/responses"
            data = {
                "id": "r",
                "object": "response",
                "created_at": 1,
                "model": "gpt-4o",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
            }
        else:
            assert request.url.path == "/v1/messages"
            data = {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        return httpx.Response(200, json=data)

    def forbid_network(*args, **kwargs):
        raise AssertionError("Test escaped mock HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbid_network)
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    params = {} if choice is None else {"tool_choice": copy.deepcopy(choice)}
    original_params = copy.deepcopy(params)
    cls = ResponsesClient if style == "responses" else CompletionClient
    client = cls(
        "openai/gpt-4o" if style == "responses" else "anthropic/claude-sonnet-4-5",
        api_base="https://models.example/v1" if style == "responses" else "https://models.example",
        api_key="test-key",
        max_tokens=16,
        **({"prompt_cache_key": "parent-shard"} if style == "responses" else {}),
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        **(params if configured else {}),
    )
    config = copy.deepcopy(client.config)
    messages = [
        {"role": "system", "content": "Stable instructions"},
        {"role": "user", "content": "Stable history"},
        CacheBoundary(),
        {"role": "user", "content": "Live suffix"},
    ]
    before = copy.deepcopy(messages)

    def lookup() -> str:
        raise AssertionError("Never execute provider tools")

    tools = [Tool(name="lookup", description="Look up a value", callable=lookup)]
    try:
        call = client.acall if is_async else client.call
        # Parent baseline and explicit choice must have identical tools/cache input.
        baseline = call(messages, tools=tools, tool_choice="auto")
        if is_async:
            await baseline
        result = call(messages, tools=tools, **({} if configured else params))
        if is_async:
            await result
        assert len(bodies) == 2
        body = bodies[1]
        if style == "responses":
            assert body["tool_choice"] == ("auto" if choice is None else choice)
            assert body["prompt_cache_key"] == "parent-shard"
            assert body["prompt_cache_options"]["mode"] == "explicit"
            assert body["parallel_tool_calls"] is False
            assert body["tools"][0]["name"] == "lookup"
            assert "prompt_cache_breakpoint" in json.dumps(body["input"])
        else:
            if choice is None or choice == "auto":
                assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
            elif isinstance(choice, dict):
                assert body["tool_choice"] == {
                    "type": "tool",
                    "name": "lookup",
                    "disable_parallel_tool_use": True,
                }
            elif choice == "required":
                assert body["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
            else:
                assert body["tool_choice"] == {"type": "none"}
            assert body["tools"][0]["name"] == "lookup"
            assert "cache_control" in json.dumps(body["messages"])
        assert {k: v for k, v in bodies[0].items() if k != "tool_choice"} == {
            k: v for k, v in body.items() if k != "tool_choice"
        }
        assert messages == before
        assert params == original_params
        assert client.config == config
    finally:
        await client.aclose()
