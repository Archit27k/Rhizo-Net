import sqlite3

class DBLogger:

    def __init__(self):

        self.connection = sqlite3.connect(
            "../../logs/rhizonet.db"
        )

        self.cursor = self.connection.cursor()

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            diagnosis TEXT,
            confidence REAL,
            edge_density REAL
        )
        """)

    def insert(
        self,
        diagnosis,
        confidence,
        edge_density
    ):

        self.cursor.execute("""
        INSERT INTO logs (
            diagnosis,
            confidence,
            edge_density
        )
        VALUES (?, ?, ?)
        """, (
            diagnosis,
            confidence,
            edge_density
        ))

        self.connection.commit()