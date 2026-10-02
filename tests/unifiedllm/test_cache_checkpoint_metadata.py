# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderer-owned neutral intent is consumed only after provider projection."""

import json
from unittest.mock import patch

import pytest

from nooa.context_blocks.formatter import OpenAIProviderFormatter
from nooa.context_blocks.models import BlockMetadata, RenderedMessage, ResolvedBlock, Role
from nooa.context_blocks.renderers.cached import CachedBlockFormatter
from nooa.llm_types import AssistantReasoning, AssistantText, CacheBoundary, LLMResponse
from nooa.unifiedllm import FakeLLMClient, ResponsesClient
from nooa.unifiedllm.cache_policy import apply_cache_policy
from nooa.unifiedllm.replay_state import prepare_chat_messages

HINT = "nooa_cache_checkpoint"


def _blocks(*, suffix=True):
    blocks = [ResolvedBlock(key="stable", content="stable", metadata=BlockMetadata(static=True))]
    if suffix:
        blocks.append(ResolvedBlock(key="live", content="live"))
    return blocks


def _prepare(messages, mapping="auto", instructions=None):
    with patch(
        "nooa.unifiedllm.cache_policy._mark_responses_cache_breakpoint",
        side_effect=AssertionError("declared renderer plan must bypass legacy inference"),
    ):
        return apply_cache_policy(messages, mapping, responses=True, instructions=instructions)


def test_renderer_declares_neutral_candidates_only_with_suffix():
    formatter = CachedBlockFormatter()
    neutral = formatter.format(_blocks())
    assert neutral[0].cache_checkpoint is True
    assert neutral[-2].replay_message.checkpoints_declared is True
    assert neutral[-1].cache_checkpoint is False
    assert RenderedMessage.model_validate_json(neutral[0].model_dump_json()).cache_checkpoint
    assert all(not msg.cache_checkpoint for msg in formatter.format(_blocks(suffix=False)))
    assert not any(isinstance(msg.replay_message, CacheBoundary) for msg in formatter.format([]))


@pytest.mark.parametrize("count", [1, 79, 80, 81, 150])
def test_latest80_rolls_over_eligible_projected_inputs_not_neutral_assistants(count):
    neutral = []
    for i in range(count):
        neutral.append(RenderedMessage(role=Role.USER, content=str(i), cache_checkpoint=True))
        # Hundreds of neutral/native assistant outputs must not consume the window.
        for _ in range(2):
            turn = LLMResponse(parts=(AssistantText(text="answer"),))
            neutral.append(
                RenderedMessage(
                    role=Role.ASSISTANT,
                    content=turn.content,
                    replay_message=turn,
                    cache_checkpoint=True,
                )
            )
    neutral.extend(
        [
            RenderedMessage(
                role=Role.METADATA, replay_message=CacheBoundary(checkpoints_declared=True)
            ),
            RenderedMessage(role=Role.USER, content="live"),
        ]
    )
    original = OpenAIProviderFormatter().format(neutral)
    with ResponsesClient("openai/gpt-5.6") as client:
        projected, instructions = client._transform_messages(original)
    wire, _, enabled = _prepare(projected, instructions=instructions)
    marked = [
        item["content"][0]["text"] for item in wire if "prompt_cache_breakpoint" in repr(item)
    ]
    assert marked == [str(i) for i in range(max(0, count - 80), count)]
    assert enabled
    assert HINT not in json.dumps(wire)
    assert "prompt_cache_breakpoint" not in repr(wire[-1])
    assert HINT in original[0]  # projection never mutates caller intent


@pytest.mark.parametrize("mapping", [None, "auto", "openai"])
def test_declared_empty_plan_does_not_infer_unhinted_prefix_or_instructions(mapping):
    wire, instructions, enabled = _prepare(
        [
            {"role": "user", "content": "unhinted"},
            CacheBoundary(checkpoints_declared=True),
            {"role": "user", "content": "live", HINT: True},
        ],
        mapping,
        instructions="unhinted instructions",
    )
    assert "prompt_cache_breakpoint" not in repr(wire)
    assert instructions == "unhinted instructions"
    assert enabled is (mapping == "openai")
    assert HINT not in repr(wire)


def test_declared_leading_system_fallback_and_no_candidate():
    rendered = OpenAIProviderFormatter().format(CachedBlockFormatter().format(_blocks()))
    with ResponsesClient("openai/gpt-5.6") as client:
        projected, instructions = client._transform_messages(rendered)
    wire, instructions, enabled = _prepare(projected, instructions=instructions)
    assert enabled and instructions is None
    assert wire[0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert "live" in wire[-1]["content"][0]["text"]
    assert "prompt_cache_breakpoint" not in repr(wire[-1])
    wire, _, enabled = _prepare([CacheBoundary(checkpoints_declared=True)])
    assert wire == [] and not enabled


def test_legacy_boundary_and_plain_callers_still_infer():
    for messages in (
        [{"role": "system", "content": "stable"}, {"role": "user", "content": "live"}],
        [
            {"role": "user", "content": "stable"},
            CacheBoundary(),
            {"role": "user", "content": "live"},
        ],
    ):
        wire, _, enabled = apply_cache_policy(messages, "openai", responses=True)
        assert enabled and "prompt_cache_breakpoint" in repr(wire[0])
        assert "prompt_cache_breakpoint" not in repr(wire[-1])


def test_native_metadata_copy_preserves_parts_scope_raw_and_original():
    raw = object()
    turn = LLMResponse(
        parts=(
            AssistantReasoning(native={"type": "reasoning", "encrypted_content": "opaque"}),
            AssistantText(text="answer"),
        ),
        replay_scope="responses:openai:test",
        raw_response=raw,
        metadata={"source": "event"},
    )
    copied = OpenAIProviderFormatter().format(
        [
            RenderedMessage(
                role=Role.ASSISTANT,
                content=turn.content,
                replay_message=turn,
                cache_checkpoint=True,
            ),
        ]
    )[0]
    assert copied is not turn and copied.parts is turn.parts
    assert copied.replay_scope == turn.replay_scope and copied.raw_response is raw
    assert copied.metadata is not turn.metadata
    assert copied.metadata == {**turn.metadata, HINT: True}
    assert HINT not in turn.metadata and HINT not in copied.public_message()
    with ResponsesClient("openai/gpt-5.6") as client:
        projected, instructions = client._transform_messages(
            [
                copied,
                CacheBoundary(checkpoints_declared=True),
                {"role": "user", "content": "live"},
            ],
            turn.replay_scope,
        )
    wire, _, enabled = _prepare(projected, instructions=instructions)
    assert not enabled  # native reasoning/output is not an input endpoint
    assert wire[0]["encrypted_content"] == "opaque" and HINT not in repr(wire)


def test_anthropic_latest_eligible_and_other_chat_routes_strip_intent():
    turn = LLMResponse(parts=(AssistantText(text="latest answer"),), metadata={HINT: True})
    messages = [
        {"role": "user", "content": "stable", HINT: True},
        turn,
        CacheBoundary(checkpoints_declared=True),
        {"role": "user", "content": "live"},
    ]
    projected = prepare_chat_messages(messages, None, anthropic_cache_marking=True)
    wire, _, _ = apply_cache_policy(projected, "anthropic", responses=False)
    assert "cache_control" not in repr(wire[0])
    assert wire[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in repr(wire[-1]) and HINT not in repr(wire)
    client = FakeLLMClient()
    client.call(messages)
    assert HINT not in repr(client.last_messages)


def test_relay_json_omits_hints_restores_unchanged_and_demotes_edits():
    from nooa.nemo_relay_middleware import _reconcile_messages, _relay_message

    turn = LLMResponse(parts=(AssistantText(text="answer"),), metadata={HINT: True})
    messages = [
        {"role": "user", "content": "stable", HINT: True},
        turn,
        CacheBoundary(checkpoints_declared=True),
    ]
    public = [_relay_message(msg) for msg in messages]
    assert HINT not in json.dumps(public) and "checkpoints_declared" not in repr(public)
    restored = _reconcile_messages(messages, public)
    assert all(a is b for a, b in zip(restored, messages, strict=True))
    public[0]["content"] = "edited"
    public[1]["content"] = "edited answer"
    edited = _reconcile_messages(messages, public)
    assert edited[:2] == public[:2] and HINT not in repr(edited[:2])
    wire, _, enabled = _prepare(edited)
    assert not enabled and "prompt_cache_breakpoint" not in repr(wire)


def test_request_checkpoint_metadata_survives_sqlite_snapshot(tmp_path):
    from nooa import Agent
    from nooa.storage import SQLiteStorageManager

    turn = LLMResponse(
        parts=(
            AssistantReasoning(native={"type": "reasoning", "encrypted_content": "opaque"}),
            AssistantText(text="answer"),
        ),
        replay_scope="responses:openai:test",
        metadata={HINT: True},
        raw_response=object(),
    )
    agent = Agent(llm=FakeLLMClient())
    agent.request_messages = [
        {"role": "user", "content": "stable", HINT: True},
        turn,
        CacheBoundary(checkpoints_declared=True),
        {"role": "user", "content": "live"},
    ]
    restored = Agent(llm=FakeLLMClient())
    with SQLiteStorageManager(tmp_path / "snapshot.db") as storage:
        snapshot_id = storage.save_snapshot(agent)
        storage.restore_snapshot(snapshot_id, restored)
    messages = restored.request_messages
    assert messages[0][HINT] is True
    assert isinstance(messages[1], LLMResponse) and messages[1].metadata[HINT] is True
    assert messages[1].parts[0].native["encrypted_content"] == "opaque"
    assert messages[1].replay_scope == turn.replay_scope and messages[1].raw_response is None
    assert isinstance(messages[2], CacheBoundary) and messages[2].checkpoints_declared
    with ResponsesClient("openai/gpt-5.6") as client:
        projected, instructions = client._transform_messages(messages, turn.replay_scope)
    wire, _, enabled = _prepare(projected, instructions=instructions)
    assert enabled and "prompt_cache_breakpoint" in repr(wire[0])
    assert HINT not in repr(wire) and "prompt_cache_breakpoint" not in repr(wire[-1])


def test_tool_results_project_to_eligible_endpoints_before_windowing():
    neutral = [RenderedMessage(role=Role.SYSTEM, content="stable", cache_checkpoint=True)]
    for index in range(85):
        neutral.append(
            RenderedMessage(
                role=Role.TOOL,
                content=str(index),
                tool_call_id=f"call_{index}",
                cache_checkpoint=True,
            )
        )
        neutral.append(
            RenderedMessage(role=Role.ASSISTANT, content="answer", cache_checkpoint=True)
        )
    neutral.append(
        RenderedMessage(
            role=Role.METADATA,
            replay_message=CacheBoundary(checkpoints_declared=True),
        )
    )
    with ResponsesClient("openai/gpt-5.6") as client:
        projected, instructions = client._transform_messages(
            OpenAIProviderFormatter().format(neutral)
        )
    wire, returned_instructions, _ = _prepare(projected, instructions=instructions)
    marked = [item for item in wire if "prompt_cache_breakpoint" in repr(item)]
    assert len(marked) == 80 and all(item["type"] == "function_call_output" for item in marked)
    assert [item["output"][0]["text"] for item in marked] == [str(i) for i in range(5, 85)]
    assert returned_instructions == "stable"  # no fallback when input candidates exist
    assert HINT not in repr(wire)


def test_mixed_leading_systems_do_not_promote_unhinted_instructions():
    with ResponsesClient("openai/gpt-5.6") as client:
        projected, instructions = client._transform_messages(
            [
                {"role": "system", "content": "stable", HINT: True},
                {"role": "system", "content": "relay edited instructions"},
                CacheBoundary(checkpoints_declared=True),
                {"role": "user", "content": "live"},
            ]
        )
    wire, instructions, enabled = _prepare(projected, instructions=instructions)
    assert not enabled
    assert instructions == "stable\n\nrelay edited instructions"
    assert "prompt_cache_breakpoint" not in repr(wire)


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_leading_system_order_and_relay_edit_disable_aggregated_fallback(reverse):
    from nooa.nemo_relay_middleware import _reconcile_messages, _relay_message

    systems = [
        {"role": "system", "content": "first", HINT: True},
        {"role": "system", "content": "second", HINT: True},
    ]
    public = [_relay_message(msg) for msg in systems]
    public[0 if reverse else 1]["content"] = "edited"
    messages = _reconcile_messages(systems, public)
    with ResponsesClient("openai/gpt-5.6") as client:
        projected, instructions = client._transform_messages(
            [
                *messages,
                CacheBoundary(checkpoints_declared=True),
                {"role": "user", "content": "live"},
            ]
        )
    wire, _, enabled = _prepare(projected, instructions=instructions)
    assert not enabled and "prompt_cache_breakpoint" not in repr(wire)
    assert systems[0]["content"] == "first" and systems[1]["content"] == "second"


@pytest.mark.parametrize("field", ["content", "output"])
def test_nested_relay_edits_demote_intent_without_mutating_original(field):
    from nooa.nemo_relay_middleware import _reconcile_messages, _relay_message

    message = {"role": "user", field: [{"type": "input_text", "text": "stable"}], HINT: True}
    public = _relay_message(message)
    public[field][0]["text"] = "intercepted"
    restored = _reconcile_messages([message], [public])
    assert restored[0] is public and HINT not in restored[0]
    assert message[field][0]["text"] == "stable"


def test_declared_empty_anthropic_plan_never_infers_or_marks_suffix():
    messages = [
        {"role": "user", "content": "unhinted"},
        CacheBoundary(checkpoints_declared=True),
        {"role": "user", "content": "live", HINT: True},
    ]
    with patch(
        "nooa.unifiedllm.cache_policy._mark_anthropic",
        side_effect=AssertionError("no pre-boundary candidate"),
    ):
        wire, _, _ = apply_cache_policy(messages, "anthropic", responses=False)
    assert "cache_control" not in repr(wire) and HINT not in repr(wire)


@pytest.mark.parametrize("media", ["input_image", "input_file"])
def test_declared_multimodal_and_ineligible_blocks_only_count_eligible_endpoints(media):
    content = [
        {"type": media, "image_url": "https://example.test/image"}
        if media == "input_image"
        else {"type": media, "file_id": "file-test"}
    ]
    messages = []
    for i in range(85):
        messages.append({"role": "user", "content": content, HINT: True, "candidate": i})
        messages.append(
            {"role": "user", "content": [{"type": "output_text", "text": "skip"}], HINT: True}
        )
    messages.append(CacheBoundary(checkpoints_declared=True))
    wire, _, _ = _prepare(messages)
    marked = [msg for msg in wire if "prompt_cache_breakpoint" in repr(msg)]
    assert [msg["candidate"] for msg in marked] == list(range(5, 85))
    assert all(msg["content"][0]["type"] == media for msg in marked)
    assert HINT not in repr(wire) and "prompt_cache_breakpoint" not in repr(content)
