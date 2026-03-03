"""
SQLite service for client, complaint, and followup persistence.

DB file: showroom-agent/data/ibyco.db  (auto-created on first use)
Schema:  ../../schema.sql  (DDL only — CREATE TABLE statements)
"""
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "ibyco.db"
SCHEMA_PATH = Path(__file__).parent.parent.parent / "schema.sql"

_conn: sqlite3.Connection = None


# ---------------------------------------------------------------------------
# Connection & initialisation
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection) -> None:
    """Run only the CREATE TABLE statements from schema.sql."""
    if SCHEMA_PATH.exists():
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        stmts = [
            s.strip() for s in sql.split(";")
            if s.strip().upper().startswith("CREATE")
        ]
        cursor = conn.cursor()
        for stmt in stmts:
            cursor.execute(stmt)
        conn.commit()
    else:
        # Minimal fallback DDL
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS clients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_number VARCHAR(30) NOT NULL UNIQUE,
                chat_summary TEXT,
                last_user_reply TEXT,
                last_bot_reply TEXT,
                last_bot_reply_type VARCHAR(50),
                last_user_message_at DATETIME,
                last_bot_message_at DATETIME,
                info TEXT,
                has_purchased BOOLEAN DEFAULT 0,
                purchase_date DATETIME,
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id INTEGER,
                message_text TEXT,
                is_resolved BOOLEAN DEFAULT 0,
                resolved_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (client_id) REFERENCES clients(id)
            );
        """)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_client(conn: sqlite3.Connection, phone_number: str) -> int:
    """Return the client id, creating the row if it doesn't exist."""
    row = conn.execute(
        "SELECT id FROM clients WHERE phone_number = ?", (phone_number,)
    ).fetchone()
    if row:
        return row[0]
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO clients (phone_number, created_at) VALUES (?, ?)",
        (phone_number, now),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upsert_client(
    phone_number: str,
    name: str = None,
    last_user_reply: str = None,
    last_bot_reply: str = None,
    last_bot_reply_type: str = None,
    has_purchased: bool = False,
) -> int:
    """Insert or update a client record. Returns the client id."""
    conn = _get_conn()
    now = datetime.utcnow().isoformat()
    client_id = _ensure_client(conn, phone_number)

    fields: dict = {
        "last_user_message_at": now,
        "last_bot_message_at": now,
    }
    if name:
        fields["info"] = name
    if last_user_reply:
        fields["last_user_reply"] = last_user_reply
    if last_bot_reply:
        fields["last_bot_reply"] = last_bot_reply
    if last_bot_reply_type:
        fields["last_bot_reply_type"] = last_bot_reply_type
    if has_purchased:
        fields["has_purchased"] = 1
        fields["purchase_date"] = now

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE clients SET {set_clause} WHERE id = ?",
        [*fields.values(), client_id],
    )
    conn.commit()
    return client_id


def save_complaint(phone_number: str, message_text: str) -> int:
    """Save a complaint linked to a client. Returns the complaint id."""
    conn = _get_conn()
    client_id = _ensure_client(conn, phone_number)
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO complaints (client_id, message_text, created_at) VALUES (?, ?, ?)",
        (client_id, message_text, now),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]
