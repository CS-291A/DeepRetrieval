import sqlite3
from contextlib import closing
from func_timeout import func_timeout, FunctionTimedOut



class SpiderDatabaseSearcher:
    def __init__(self):
        self.timeout = 30

    def search(self, sql_query: str, db_path: str, db_id: str | None=None):
        try:
            results = func_timeout(self.timeout, self._search, args=(sql_query, db_path, db_id))

        except FunctionTimedOut:
            # None (not []) so a timed-out query can never equal an empty result set
            results = None

        return results

    def _search(self, sql_query: str, db_path: str, db_id: str | None=None):
        # Connect to the database
        with closing(sqlite3.connect(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
        return results
