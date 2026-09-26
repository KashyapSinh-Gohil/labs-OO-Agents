# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLiteStorageManager public options: save_snapshot_json, must_exist, journal_mode."""

import json
import sqlite3
import threading
from unittest import mock

import pytest

from nooa.storage import SQLiteStorageManager
from nooa.storage import sqlite as sqlite_module


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


def _journal_mode(storage):
    return storage._conn.execute("PRAGMA journal_mode").fetchone()[0]


def test_must_exist_refuses_a_missing_file_and_creates_nothing(tmp_path):
    path = tmp_path / "gone.db"
    with pytest.raises(sqlite3.OperationalError):
        SQLiteStorageManager(path, must_exist=True)
    assert not path.exists()
    # The lock was released: a normal open still works.
    with SQLiteStorageManager(path) as storage:
        storage.save_snapshot_json("{}")


def test_must_exist_opens_an_existing_file(tmp_path):
    path = tmp_path / "s.db"
    with SQLiteStorageManager(path) as storage:
        first = storage.save_snapshot_json("{}")
    with SQLiteStorageManager(path, must_exist=True) as storage:
        assert storage.get_latest_snapshot_id() == first
        storage.save_snapshot_json("{}")


def test_must_exist_is_rejected_for_memory():
    with pytest.raises(ValueError):
        SQLiteStorageManager(":memory:", must_exist=True)


def test_journal_mode_delete_and_wal_are_used_as_given(tmp_path):
    with SQLiteStorageManager(tmp_path / "d.db", journal_mode="delete") as storage:
        assert _journal_mode(storage) == "delete"
    with SQLiteStorageManager(tmp_path / "w.db", journal_mode="wal") as storage:
        assert _journal_mode(storage) == "wal"


def test_journal_mode_delete_converts_a_wal_file(tmp_path):
    path = tmp_path / "s.db"
    with SQLiteStorageManager(path, journal_mode="wal") as storage:
        storage.save_snapshot_json("{}")
    with SQLiteStorageManager(path, journal_mode="delete") as storage:
        assert _journal_mode(storage) == "delete"
        assert storage.get_latest_snapshot_id() is not None
    assert not (tmp_path / "s.db-wal").exists()


def test_journal_mode_none_keeps_the_virtiofs_detection(tmp_path):
    with mock.patch.object(sqlite_module, "_is_virtiofs", return_value=True):
        with SQLiteStorageManager(tmp_path / "v.db") as storage:
            assert _journal_mode(storage) == "delete"
    with mock.patch.object(sqlite_module, "_is_virtiofs", return_value=False):
        with SQLiteStorageManager(tmp_path / "n.db") as storage:
            assert _journal_mode(storage) == "wal"
    # An explicit mode skips detection.
    with mock.patch.object(sqlite_module, "_is_virtiofs", side_effect=AssertionError):
        with SQLiteStorageManager(tmp_path / "e.db", journal_mode="wal") as storage:
            assert _journal_mode(storage) == "wal"


def test_journal_mode_rejects_other_values(tmp_path):
    with pytest.raises(ValueError):
        SQLiteStorageManager(tmp_path / "x.db", journal_mode="truncate")  # type: ignore[arg-type]
    assert not (tmp_path / "x.db").exists()
