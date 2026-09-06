# import sqlite3
# import threading
# from config import DB_PATH

# print("DB PATH =", DB_PATH)
# print("Exists =", DB_PATH.parent.exists())


# class DatabaseManager:

#     def __init__(self):

#         self.conn = sqlite3.connect(
#             str(DB_PATH),
#             check_same_thread=False
#         )

#         self.lock = threading.RLock()

#         self.create_tables()

#     def create_tables(self):

#         with self.lock:

#             cursor = self.conn.cursor()

#             try:

#                 cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS users (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     name TEXT,
#                     email TEXT UNIQUE,
#                     password TEXT,
#                     role TEXT
#                 )
#                 """)

#                 cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS quiz_scores (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     student TEXT,
#                     subject TEXT,
#                     score INTEGER,
#                     weak_topic TEXT,
#                     date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#                 )
#                 """)

#                 cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS attendance (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     student_name TEXT,
#                     date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                     status TEXT
#                 )
#                 """)

#                 cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS study_plans (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     student TEXT,
#                     task TEXT,
#                     subject TEXT,
#                     plan_date TEXT,
#                     duration INTEGER,
#                     status TEXT DEFAULT 'Pending',
#                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#                 )
#                 """)

#                 self.conn.commit()

#             finally:
#                 cursor.close()


# db = DatabaseManager()

import sqlite3
import threading
from config import DB_PATH

print("DB PATH =", DB_PATH)
print("Exists =", DB_PATH.parent.exists())


class DatabaseManager:
    """
    Thread-safe SQLite manager.

    The old version kept one cursor globally:
        self.cursor = self.conn.cursor()

    Streamlit reruns and background workers can cause that cursor to be
    reused while it is still active. This version keeps a separate SQLite
    connection and cursor per Python thread, while preserving the existing
    db.conn / db.cursor API used by app.py and auth.py.
    """

    def __init__(self):
        self._local = threading.local()

        # Create the database/tables immediately on the main thread.
        self._get_connection()
        self.create_tables()

    def _get_connection(self):
        """Return the SQLite connection belonging to the current thread."""
        conn = getattr(self._local, "conn", None)

        if conn is None:
            conn = sqlite3.connect(
                str(DB_PATH),
                check_same_thread=False,
                timeout=30
            )

            # Helps concurrent Streamlit operations on the same DB file.
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA journal_mode = WAL")

            self._local.conn = conn

        return conn

    @property
    def conn(self):
        """Backward-compatible current-thread connection."""
        return self._get_connection()

    @property
    def cursor(self):
        """Backward-compatible current-thread cursor."""
        cursor = getattr(self._local, "cursor", None)

        if cursor is None:
            cursor = self.conn.cursor()
            self._local.cursor = cursor

        return cursor

    def create_tables(self):
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                password TEXT,
                role TEXT
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT,
                subject TEXT,
                score INTEGER,
                weak_topic TEXT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT
            )
            """)

            # Keep this table here instead of creating it again in app.py.
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS study_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT,
                task TEXT,
                subject TEXT,
                plan_date TEXT,
                duration INTEGER,
                status TEXT DEFAULT 'Pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            self.conn.commit()

        finally:
            cursor.close()

            # Do not leave a closed cursor cached for this thread.
            self._local.cursor = None

    def close(self):
        """Close the current thread's cursor and connection."""
        cursor = getattr(self._local, "cursor", None)
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
            self._local.cursor = None

        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None


# Important: create instance here
db = DatabaseManager()
