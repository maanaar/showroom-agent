"""
SQLite service for client, complaint, and followup persistence.

DB file: showroom-agent/data/ibyco.db  (auto-created on first use)
Schema:  ../../schema.sql  (DDL only — CREATE TABLE statements)
"""
import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "Ibyco" / "instance" / "Ibyco.db"
SCHEMA_PATH = Path(__file__).parent.parent / "Ibyco" / "schema.sql"

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


def get_installment_rate(months: int, down_payment_pct: float = 0) -> dict:
    """Return the installment plan matching the requested months and down payment %.

    The instalments table stores range-based plans (not per vehicle):
      - min_down_payment / max_down_payment : down payment % range (max exclusive)
      - min_months / max_months             : months range — lower exclusive, upper inclusive
                                              i.e. min_months < months <= max_months
      - percentage                          : total interest % for the plan's max_months
      - percentage_per_month                : percentage / max_months

    For "no down payment" callers pass down_payment_pct=0 (default).
    If months exceeds all plans, falls back to the widest plan for that down tier.

    Returns:
        {
          "percentage": <total interest %>,
          "percentage_per_month": <monthly interest %>,
          "max_months": <plan ceiling>,
        }
        or {} if nothing found.
    """
    conn = _get_conn()

    # Range match: lower bound exclusive, upper bound inclusive → no boundary ambiguity
    row = conn.execute(
        "SELECT percentage, percentage_per_month, max_months "
        "FROM instalments "
        "WHERE min_down_payment <= ? AND max_down_payment > ? "
        "  AND min_months < ? AND max_months >= ? "
        "ORDER BY max_months ASC "
        "LIMIT 1",
        (down_payment_pct, down_payment_pct, months, months),
    ).fetchone()

    if not row:
        # Fallback: widest plan available for this down payment tier
        row = conn.execute(
            "SELECT percentage, percentage_per_month, max_months "
            "FROM instalments "
            "WHERE min_down_payment <= ? AND max_down_payment > ? "
            "ORDER BY max_months DESC "
            "LIMIT 1",
            (down_payment_pct, down_payment_pct),
        ).fetchone()

    if not row:
        return {}

    return {
        "percentage": row["percentage"],
        "percentage_per_month": row["percentage_per_month"],
        "max_months": row["max_months"],
    }


def update_client_turn(
    phone_number: str,
    user_message: str,
    bot_response: str,
    intent: str = None,
    filters: dict = None,
    lead: dict = None,
) -> None:
    """Persist each conversation turn: update last replies and append a structured
    summary line to clients.chat_summary for staff visibility."""
    conn = _get_conn()
    _ensure_client(conn, phone_number)
    now = datetime.utcnow()
    now_str = now.isoformat()
    date_str = now.strftime("%Y-%m-%d %H:%M")

    # Build a compact summary line from structured state
    filters = filters or {}
    lead = lead or {}
    parts = [f"[{date_str}] النية: {intent or '—'}"]
    if filters.get("vehicle_name"):
        parts.append(f"الموديل: {filters['vehicle_name']}")
    if filters.get("company"):
        parts.append(f"الشركة: {filters['company']}")
    if filters.get("max_price"):
        parts.append(f"أقصى سعر: {filters['max_price']:,}ج")
    if filters.get("down_payment") is not None:
        parts.append(f"مقدم: {filters['down_payment']:,}ج")
    if filters.get("months"):
        parts.append(f"أشهر: {filters['months']}")
    if filters.get("max_installment_12"):
        parts.append(f"قسط مطلوب: {filters['max_installment_12']:,}ج/شهر")
    if lead.get("name"):
        parts.append(f"الاسم: {lead['name']}")
    if lead.get("phone"):
        parts.append(f"التليفون: {lead['phone']}")
    # Truncate user message for readability
    short_msg = (user_message[:80] + "...") if len(user_message) > 80 else user_message
    parts.append(f'رسالة: "{short_msg}"')

    summary_line = " | ".join(parts)

    # Append to existing summary (keep rolling log)
    existing = conn.execute(
        "SELECT chat_summary FROM clients WHERE phone_number = ?", (phone_number,)
    ).fetchone()
    old_summary = (existing["chat_summary"] or "") if existing else ""
    new_summary = (old_summary + "\n" + summary_line).strip()

    # Truncate if too long (keep last 4000 chars)
    if len(new_summary) > 4000:
        new_summary = new_summary[-4000:]

    conn.execute(
        """UPDATE clients
           SET chat_summary = ?, last_user_reply = ?, last_bot_reply = ?,
               last_user_message_at = ?, last_bot_message_at = ?
           WHERE phone_number = ?""",
        (new_summary, user_message[:500], bot_response[:500],
         now_str, now_str, phone_number),
    )
    conn.commit()


def save_booking(phone_number: str, name: str = None, vehicle_interest: str = None) -> int:
    """Save or update a booking/appointment request. Returns the client id."""
    conn = _get_conn()
    client_id = _ensure_client(conn, phone_number)
    now = datetime.utcnow().isoformat()
    fields = {"last_user_message_at": now, "last_bot_message_at": now, "last_bot_reply_type": "booking"}
    if name:
        fields["info"] = name
    if vehicle_interest:
        fields["chat_summary"] = f"حجز موعد — اهتمام بـ: {vehicle_interest}"
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE clients SET {set_clause} WHERE id = ?", [*fields.values(), client_id])
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
