# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce prefill truncation with a real LLM.

uv run python examples/quickstart/17_prefill_truncation.py         # truncated + warning
uv run python examples/quickstart/17_prefill_truncation.py 90000   # complete after fix

On the old runtime, even 90000 loses the middle marker. The original argument
remains intact in the REPL in both cases. Uses the quickstart's configured LLM.
Add --internal to use NVIDIA_INFERENCE_API_KEY with the company gateway.
"""

from nooa import Agent, CodeActStrategy, hidden, strategy
from nooa.config.truncation_config import FormatConfig, TruncationConfig

with hidden:
    import argparse
    import os
    import secrets

    from nooa.events import PythonOutput
    from nooa.unifiedllm.registry import get_llm_client
    from nooa.util.quickstart import autorun, llm


@autorun
async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("max_string", type=int, nargs="?", default=2000)
    parser.add_argument(
        "--internal", action="store_true", help="Use NVIDIA's company inference gateway"
    )
    args = parser.parse_args()
    limit = args.max_string
    client = llm
    if args.internal:
        client = get_llm_client(
            "openai/nvidia/nvidia/nemotron-3.5-lightning",
            api_base="https://inference-api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_INFERENCE_API_KEY"],
        )

    class MinimalAgent(Agent, llm=client):
        @strategy(
            CodeActStrategy(),
            truncation=TruncationConfig(prefill_format=FormatConfig(max_string=limit)),
        )
        async def measure(self, data: str) -> int:
            """Return len(data), using Python to measure the complete variable."""
            ...

    marker = "MIDDLE_" + secrets.token_hex(8)
    data = "A" * 1500 + marker + "B" * 1500
    agent = MinimalAgent()
    try:
        result = await agent.measure(data)
        prefill = "\n".join(
            event.stdout
            for event in agent.event_manager.all_events()
            if isinstance(event, PythonOutput) and event.metadata.get("prefill")
        )
        print(f"Requested max_string: {limit}")
        print(f"Middle marker present in actual prefill: {marker in prefill}")
        print(f"Full input length: {len(data)}; agent measured: {result}")
    finally:
        await agent.aclose()
