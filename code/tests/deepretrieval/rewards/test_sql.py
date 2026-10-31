"""A6, D30, G5 (approved change): SQL executors and the bird / spider rewards."""
import logging
import sqlite3

import pytest
from func_timeout import FunctionTimedOut

from src.sql import bird as bird_sql
from src.sql import spider as spider_sql

SLOW_SQL = "SELECT x FROM t WHERE x = 'slow'"  # the fake timer times this query out
EMPTY_SQL = "SELECT x FROM t WHERE 0"
ALL_SQL = "SELECT x FROM t"
EXECUTORS = [(bird_sql, bird_sql.BirdDatabaseSearcher), (spider_sql, spider_sql.SpiderDatabaseSearcher)]
# (module, score when executed results differ)
SQL_REWARDS = [
    ("verl.utils.reward_score.bird", 0.3),
    ("verl.utils.reward_score.bird_no_reason", 0.3),
    ("verl.utils.reward_score.spider", 0),
    ("verl.utils.reward_score.spider_no_reason", 0),
]


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "fixture.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (x TEXT)")
        conn.executemany("INSERT INTO t VALUES (?)", [("a",), ("b",)])
    return str(path)


def _time_out_slow_sql(timeout, func, args):
    if args[0] == SLOW_SQL:
        raise FunctionTimedOut("fake timeout")
    return func(*args)


@pytest.fixture
def fake_timer(monkeypatch):
    for module, _ in EXECUTORS:
        monkeypatch.setattr(module, "func_timeout", _time_out_slow_sql)


_REAL_CONNECT = sqlite3.connect


class RecordingConnection(sqlite3.Connection):
    """Remembers close(); the executor runs in a func_timeout thread, so the test cannot query it."""

    closed = False

    def close(self):
        self.closed = True
        super().close()


class ConnectionRecorder:
    def __init__(self):
        self.connections = []

    def __call__(self, *args, **kwargs):
        conn = _REAL_CONNECT(*args, factory=RecordingConnection, **kwargs)
        self.connections.append(conn)
        return conn


# ---------------------------------------------------------------- src/sql executors


@pytest.mark.parametrize(("module", "searcher_cls"), EXECUTORS)
def test_executor_returns_rows(db_path, module, searcher_cls):
    assert searcher_cls().search(ALL_SQL, db_path) == [("a",), ("b",)]


@pytest.mark.parametrize(("module", "searcher_cls"), EXECUTORS)
def test_executor_returns_none_on_timeout(db_path, fake_timer, module, searcher_cls):
    assert searcher_cls().search(SLOW_SQL, db_path) is None


@pytest.mark.parametrize(("module", "searcher_cls"), EXECUTORS)
def test_executor_closes_connection_after_query(db_path, monkeypatch, module, searcher_cls):
    recorder = ConnectionRecorder()
    monkeypatch.setattr(module.sqlite3, "connect", recorder)

    searcher_cls().search(ALL_SQL, db_path)

    assert [conn.closed for conn in recorder.connections] == [True]


@pytest.mark.parametrize(("module", "searcher_cls"), EXECUTORS)
def test_executor_closes_connection_after_failing_query(db_path, monkeypatch, module, searcher_cls):
    recorder = ConnectionRecorder()
    monkeypatch.setattr(module.sqlite3, "connect", recorder)

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        searcher_cls().search("SELECT * FROM missing", db_path)

    assert [conn.closed for conn in recorder.connections] == [True]


# ---------------------------------------------------------------- rewards


@pytest.mark.parametrize(("module_name", "mismatch_score"), SQL_REWARDS)
def test_sql_reward_reuses_one_searcher(load_reward_module, module_name, mismatch_score):
    module = load_reward_module(module_name)

    assert module.get_searcher() is module.get_searcher()


@pytest.mark.parametrize(("module_name", "mismatch_score"), SQL_REWARDS)
def test_sql_reward_scores_matching_results_as_one(load_reward_module, db_path, module_name, mismatch_score):
    module = load_reward_module(module_name)

    assert module.calculate_answer_score("SELECT x FROM t ORDER BY x DESC", ALL_SQL, db_path) == 1


@pytest.mark.parametrize(("module_name", "mismatch_score"), SQL_REWARDS)
def test_sql_reward_scores_different_results_as_mismatch(load_reward_module, db_path, module_name, mismatch_score):
    module = load_reward_module(module_name)

    assert module.calculate_answer_score(EMPTY_SQL, ALL_SQL, db_path) == mismatch_score


@pytest.mark.parametrize(("module_name", "mismatch_score"), SQL_REWARDS)
def test_sql_reward_timed_out_prediction_does_not_match_empty_gold(load_reward_module, db_path, fake_timer,
                                                                   module_name, mismatch_score):
    module = load_reward_module(module_name)

    assert module.calculate_answer_score(SLOW_SQL, EMPTY_SQL, db_path) == mismatch_score


@pytest.mark.parametrize(("module_name", "mismatch_score"), SQL_REWARDS)
def test_sql_reward_timed_out_gold_scores_zero(load_reward_module, db_path, fake_timer, module_name,
                                               mismatch_score):
    module = load_reward_module(module_name)

    assert module.calculate_answer_score(SLOW_SQL, SLOW_SQL, db_path) == 0


@pytest.mark.parametrize(("module_name", "mismatch_score"), SQL_REWARDS)
def test_sql_reward_timed_out_gold_is_logged_at_warning(load_reward_module, db_path, fake_timer, caplog, module_name,
                                                        mismatch_score):
    module = load_reward_module(module_name)

    with caplog.at_level(logging.DEBUG, logger=module_name):
        module.calculate_answer_score(EMPTY_SQL, SLOW_SQL, db_path)

    assert [r.levelno for r in caplog.records
            if r.name == module_name and f"Gold SQL timed out on {db_path}" in r.getMessage()] == [logging.WARNING]


@pytest.mark.parametrize(("module_name", "mismatch_score"), SQL_REWARDS)
@pytest.mark.parametrize("answer_text", ['["sql"]', '"sql"', "5", "null"])
def test_sql_reward_rejects_non_object_json_as_format_error(load_reward_module, db_path, module_name,
                                                            mismatch_score, answer_text):
    module = load_reward_module(module_name)
    think = "" if "no_reason" in module_name else "<think>t</think>"
    response = f"<|im_start|>assistant {think}<answer>{answer_text}</answer>"

    score = module.compute_score(response, {"target": ALL_SQL}, "bird", db_path)

    assert score == -2
