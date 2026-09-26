# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLiteStorageManager: save_snapshot_json from a worker thread."""

import json
import sqlite3
import threading

import pytest

from nooa.storage import SQLiteStorageManager


def _rows(path):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT snapshot_id, created_at, data FROM snapshots ORDER BY created_at"
        ).fetchall()
    finally:
        conn.close()


def test_save_snapshot_json_writes_the_given_blob(tmp_path):
    path = tmp_path / "s.db"
    with SQLiteStorageManager(path) as storage:
        blob = json.dumps({"a": 1})
        snapshot_id = storage.save_snapshot_json(blob)
        assert storage.get_latest_snapshot_id() == snapshot_id
        pinned = storage.save_snapshot_json(
            blob, snapshot_id="fixed", created_at="2999-01-01T00:00:00+00:00"
        )
        assert pinned == "fixed"
        assert storage.get_latest_snapshot_id() == "fixed"
    rows = _rows(path)
    assert [r[0] for r in rows] == [snapshot_id, "fixed"]
    assert rows[0][2] == blob


def test_save_snapshot_json_can_run_in_a_worker_thread(tmp_path):
    path = tmp_path / "s.db"
    with SQLiteStorageManager(path, check_same_thread=False) as storage:
        results: list[str] = []
        errors: list[BaseException] = []

        def work(n: int) -> None:
            try:
                results.append(storage.save_snapshot_json(json.dumps({"n": n})))
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(set(results)) == 8
    assert len(_rows(path)) == 8


def test_save_snapshot_json_rejects_a_non_json_blob(tmp_path):
    with SQLiteStorageManager(tmp_path / "s.db") as storage:
        with pytest.raises(ValueError):
            storage.save_snapshot_json("not json {")
        assert storage.get_latest_snapshot_id() is None
