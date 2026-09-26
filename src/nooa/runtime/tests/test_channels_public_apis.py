# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public core APIs added for issue #318.

These accessors replace TUI reads of ``Channel``/``QueueManager``
private members. Each test asserts the public API reproduces the exact
behaviour the TUI previously reimplemented inline.
"""

from __future__ import annotations

import asyncio

import pytest

from nooa.runtime.channels import Channel, QueueManager

# ---------------------------------------------------------------------------
# Channel.drain
# ---------------------------------------------------------------------------


def test_drain_returns_items_fifo_and_empties():
    q: Channel[str] = Channel("q", "queue")
    q.put("a")
    q.put("b")
    q.put("c")
    assert q.drain() == ["a", "b", "c"]
    assert q.qsize() == 0
    assert q.drain() == []  # already drained


def test_drain_fires_on_get_once_per_item():
    seen: list[str] = []
    q: Channel[str] = Channel("q", "queue", on_get=seen.append)
    q.put("x")
    q.put("y")
    drained = q.drain()
    # Same firing path as get()/race winner: on_get fires once per item,
    # in dequeue order, and the returned list matches what fired.
    assert drained == ["x", "y"]
    assert seen == ["x", "y"]


def test_drain_on_get_exception_does_not_lose_items():
    def boom(_item: str) -> None:
        raise RuntimeError("hook failed")

    q: Channel[str] = Channel("q", "queue", on_get=boom)
    q.put("a")
    q.put("b")
    # _fire_on_get swallows hook exceptions; items still drained.
    assert q.drain() == ["a", "b"]


def test_drain_event_mode_returns_empty():
    class _EM:
        def add(self, _event: object) -> None:  # pragma: no cover - not exercised
            pass

    ch: Channel[str] = Channel("e", "event", event_manager=_EM())
    assert ch.drain() == []


# ---------------------------------------------------------------------------
# QueueManager.channels / running_handles
# ---------------------------------------------------------------------------


def test_channels_snapshot_is_copy_in_registration_order():
    qm = QueueManager()
    a = qm.queue("a")
    b = qm.queue("b")
    snap = qm.channels()
    assert list(snap.keys()) == ["a", "b"]
    assert snap["a"] is a and snap["b"] is b
    # Mutating the snapshot must not touch the live registry.
    snap.clear()
    assert qm.names() == ["a", "b"]


@pytest.mark.asyncio
async def test_running_handles_filters_by_state():
    qm = QueueManager()
    qm.queue("jobs")

    async def _forever() -> None:
        await asyncio.Event().wait()

    async def _quick() -> str:
        return "done"

    h_run = qm.spawn(_forever(), channel="jobs")
    h_done = qm.spawn(_quick(), channel="jobs")
    await asyncio.sleep(0.01)  # let _quick finish

    running = qm.running_handles()
    assert h_run in running
    assert h_done not in running
    assert all(h.state == "running" for h in running)

    await qm.shutdown()


# ---------------------------------------------------------------------------
# QueueManager.set_notify_callback
# ---------------------------------------------------------------------------


def test_notify_callback_fires_before_race_pair_exists():
    """The host callback must fire even on the first put, when the internal
    notify pair has not been created yet (race() has never run). This is the
    behaviour the TUI monkey-patch relied on to start the dispatcher."""
    qm = QueueManager()
    q = qm.queue("q")
    calls: list[int] = []
    qm.set_notify_callback(lambda: calls.append(1))
    q.put("first")
    assert calls == [1]


@pytest.mark.asyncio
async def test_notify_callback_fires_after_internal_wakeup():
    qm = QueueManager()
    ev_ch = qm.queue("q")
    calls: list[int] = []
    qm.set_notify_callback(lambda: calls.append(1))

    # Prime the notify pair via a race() that we immediately satisfy.
    async def _producer() -> None:
        await asyncio.sleep(0.01)
        ev_ch.put("item")

    task = asyncio.create_task(_producer())
    result = await qm.race()
    await task
    assert result == [("q", "item")]
    # put() fired the callback in addition to waking race().
    assert calls == [1]


def test_notify_callback_none_clears():
    qm = QueueManager()
    q = qm.queue("q")
    calls: list[int] = []
    qm.set_notify_callback(lambda: calls.append(1))
    qm.set_notify_callback(None)
    q.put("x")
    assert calls == []


# ---------------------------------------------------------------------------
# Channel.set_on_discard
# ---------------------------------------------------------------------------


def test_on_discard_fires_for_items_dropped_without_a_consumer():
    seen: list[list[str]] = []
    q: Channel[str] = Channel("q", "queue")
    q.set_on_discard(seen.append)
    for item in ("a", "b", "c", "d"):
        q.put(item)
    assert q.pop_last() == "d"
    q.put("e")
    assert q.flush() == 4
    q.put("f")
    q.clear()
    q.flush()  # nothing left: no call
    assert seen == [["d"], ["a", "b", "c", "e"], ["f"]]


def test_on_discard_fires_when_the_channel_is_removed():
    seen: list[list[str]] = []
    qm = QueueManager()
    q: Channel[str] = qm.queue("q")
    q.set_on_discard(seen.append)
    q.put("x")
    qm.remove_channel("q")
    assert seen == [["x"]]


def test_on_discard_is_not_fired_for_consumed_items_and_its_errors_are_swallowed():
    def boom(_items: list[str]) -> None:
        raise RuntimeError("hook failed")

    q: Channel[str] = Channel("q", "queue")
    q.put("a")
    assert q.drain() == ["a"]
    q.set_on_discard(boom)
    q.put("b")
    assert q.flush() == 1
    assert q.qsize() == 0


# ---------------------------------------------------------------------------
# Channel.remove and the on_get / on_discard getters
# ---------------------------------------------------------------------------


def test_remove_withdraws_one_item_by_identity_without_firing_hooks():
    got: list[object] = []
    dropped: list[list[object]] = []
    q: Channel[object] = Channel("q", "queue", on_get=got.append)
    q.set_on_discard(dropped.append)
    first, second, equal_not_same = ["a"], ["b"], ["a"]
    q.put(first)
    q.put(second)
    q.put(first)

    assert q.remove(equal_not_same) is False  # equality is not enough
    assert q.remove(first) is True  # the head occurrence goes first
    assert q.snapshot() == [second, first]
    assert q.snapshot()[1] is first
    assert q.remove(first) is True
    assert q.remove(first) is False
    assert q.snapshot() == [second]
    # A withdraw is neither a consume nor a discard.
    assert got == []
    assert dropped == []


def test_remove_on_event_channel_returns_false():
    q: Channel[str] = Channel("e", "event")
    assert q.remove("x") is False


def test_hook_getters_return_current_callbacks_for_chaining():
    q: Channel[str] = Channel("q", "queue")
    assert q.on_get is None
    assert q.on_discard is None

    first: list[str] = []
    q.set_on_get(first.append)
    q.set_on_discard(lambda items: first.extend(f"x{i}" for i in items))
    assert q.on_get is not None

    previous_get, previous_discard = q.on_get, q.on_discard
    second: list[str] = []

    def chained_get(item: str) -> None:
        second.append(item)
        previous_get(item)

    def chained_discard(items: list[str]) -> None:
        second.extend(f"x{i}" for i in items)
        previous_discard(items)

    q.set_on_get(chained_get)
    q.set_on_discard(chained_discard)
    assert q.on_get is chained_get
    assert q.on_discard is chained_discard

    q.put("a")
    q.put("b")
    assert q.drain() == ["a", "b"]
    q.put("c")
    q.clear()
    assert first == ["a", "b", "xc"]
    assert second == ["a", "b", "xc"]


def test_hook_getters_are_read_only():
    q: Channel[str] = Channel("q", "queue")
    with pytest.raises(AttributeError):
        q.on_get = print  # type: ignore[misc]
