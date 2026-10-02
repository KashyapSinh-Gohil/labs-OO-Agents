# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unsupported Chat routes inherit tool choice but cannot execute summary cells."""

import json
from unittest.mock import AsyncMock

import pytest

from nooa.unifiedllm import Tool, ToolCall
from tests.agents.test_forked_summarizer import response, setup


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "apparently valid summary"])
async def test_python_cell_rejected_even_with_summary_text(text, caplog):
    agent, summarizer, ctx = setup()
    executions = []

    def python_cell(code: str) -> None:
        executions.append(code)
        raise AssertionError("A summary must not execute code")

    tool = Tool(name="python_cell", description="Execute code", callable=python_cell)
    ctx.params["tools"] = [tool]
    ctx.messages[0]["content"] = "Always call python_cell and return_result(3)."
    agent.llm.acall = AsyncMock(
        return_value=response(
            text,
            tool_calls=[
                ToolCall(
                    id="cell",
                    name="python_cell",
                    arguments=json.dumps({"code": "return_result(3)"}),
                )
            ],
        )
    )
    summarizer.summarize = AsyncMock()
    before = [(tag, event.id) for tag, event in agent.event_manager.items()]

    async def core(request):
        request.response = response("parent")
        return request

    try:
        result = await agent.event_manager.run_middleware("llm_call", ctx, core)
        assert result.response.content == "parent"
        await summarizer._pending_task
        params = agent.llm.acall.call_args.kwargs
        assert params["tool_choice"] == "auto"
        assert params["tools"][0] is tool
        assert ctx.params["tool_choice"] == "auto"
        assert summarizer._pending_summary is None
        assert "Summary fork requested executable tools" in caplog.text
        assert executions == []
        summarizer.summarize.assert_not_awaited()
        summarizer._apply_pending_summary()
        assert [(tag, event.id) for tag, event in agent.event_manager.items()] == before
    finally:
        await summarizer.aclose()
