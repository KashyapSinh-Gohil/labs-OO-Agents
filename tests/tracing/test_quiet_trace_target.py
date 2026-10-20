# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""quiet_trace_target() hides only the "OTel tracing enabled" banner."""

import nooa.tracing as tracing


class _Exporter:
    def describe(self):
        return "journal:http://viewer.test:5001"


def test_banner_prints_by_default(capsys):
    tracing._print_trace_target([_Exporter()], None)
    assert "OTel tracing enabled: journal:http://viewer.test:5001" in capsys.readouterr().out


def test_banner_is_silent_inside_quiet_block_and_restored_after(capsys):
    with tracing.quiet_trace_target():
        tracing._print_trace_target([_Exporter()], "exp")
        with tracing.quiet_trace_target():
            pass
        tracing._print_trace_target([_Exporter()], None)
    captured = capsys.readouterr()
    assert "OTel tracing" not in captured.out + captured.err
    tracing._print_trace_target([_Exporter()], None)
    assert "OTel tracing enabled" in capsys.readouterr().out


def test_quiet_block_restores_state_after_an_error(capsys):
    try:
        with tracing.quiet_trace_target():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    tracing._print_trace_target([_Exporter()], None)
    assert "OTel tracing enabled" in capsys.readouterr().out
