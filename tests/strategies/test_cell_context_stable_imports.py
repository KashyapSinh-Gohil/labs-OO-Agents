# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cell-context stub depends on the bound objects, not on live ``sys.modules``.

It renders at the front of every request. When a child agent re-imports a
library (its registry refreshes ``sys.modules``), or that entry is later
dropped, the parent's bound classes are no longer the objects in
``sys.modules``; the stub must still read the same, or the provider's prompt
cache restarts for the whole conversation.
"""

import sys
from types import ModuleType

import pytest

from nooa import CodeActV2
from nooa.config import CodeActConfig

LIB = "stable_import_lib"
LIB_SRC = "class Thing: pass\nclass Helper: pass\ndef util(): pass\n"


def _fresh_lib() -> ModuleType:
    module = ModuleType(LIB)
    exec(LIB_SRC, vars(module))
    return module


async def _render(agent_module: ModuleType) -> str:
    strategy = CodeActV2(config=CodeActConfig(prefill=None))
    agent = type("Agent", (), {})()
    agent.__class__.__module__ = agent_module.__name__
    runtime = type("Runtime", (), {"agent": agent})()
    return await strategy.python_cell_context(runtime)


@pytest.mark.asyncio
async def test_stub_is_stable_across_library_reimport_and_removal(monkeypatch):
    """Re-importing or dropping the library elsewhere does not change the stub."""
    monkeypatch.setitem(sys.modules, LIB, _fresh_lib())
    agent_module = ModuleType("stable_import_agent")
    exec(f"from {LIB} import Thing, Helper as Aid, util", vars(agent_module))
    monkeypatch.setitem(sys.modules, agent_module.__name__, agent_module)

    first = await _render(agent_module)
    assert f"from {LIB} import Helper as Aid, Thing, util" in first
    assert "Other bound names" not in first

    # Another agent's registry refreshed the library: same source, new objects.
    monkeypatch.setitem(sys.modules, LIB, _fresh_lib())
    assert sys.modules[LIB].Thing is not agent_module.Thing
    assert await _render(agent_module) == first

    # ...and later dropped it on shutdown.
    monkeypatch.delitem(sys.modules, LIB)
    assert await _render(agent_module) == first


@pytest.mark.asyncio
async def test_nested_definitions_still_use_the_plain_listing(monkeypatch):
    """Only top-level definitions get a synthesized import when unresolvable."""
    lib = _fresh_lib()
    exec("class Outer:\n    class Inner: pass\n", vars(lib))
    monkeypatch.setitem(sys.modules, LIB, lib)
    agent_module = ModuleType("stable_import_agent_nested")
    exec(
        f"from {LIB} import Outer\nInner = Outer.Inner\nfrom typing import List as Seq",
        vars(agent_module),
    )
    monkeypatch.setitem(sys.modules, agent_module.__name__, agent_module)
    monkeypatch.delitem(sys.modules, LIB)

    rendered = await _render(agent_module)
    assert f"from {LIB} import Outer" in rendered
    assert "Inner" in rendered.split("Other bound names")[-1]
