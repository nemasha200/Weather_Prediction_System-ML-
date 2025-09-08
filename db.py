# db.py
import os, sqlite3, hashlib, hmac, secrets
from contextlib import closing
from typing import Optional, Tuple, List

# --- DB path resolution ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("AUTH_DB")
if not DB_PATH:
    candidates = [
        os.path.join(BASE_DIR, "auth.db"),
        os.path.join(BASE_DIR, "..", "auth.db"),
    ]
    DB_PATH = next((p for p in candidates if os.path.exists(p)), candidates[0])

def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def ensure_schema() -> None:
    with _connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT UNIQUE COLLATE NOCASE,
                full_name  TEXT NOT NULL,
                password   TEXT NOT NULL,   -- bcrypt$... or sha256$<salt>$<hash> or legacy
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        con.commit()

# --- CRUD helpers ---
def create_user(username: str, full_name: str, password: str, scheme: str = "sha256") -> int:
    ensure_schema()
    stored = hash_password(password, scheme=scheme)
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO users (username, full_name, password) VALUES (?, ?, ?)",
            (username, full_name, stored),
        )
        con.commit()
        return int(cur.lastrowid)

def get_user(username: str) -> Optional[Tuple]:
    """Return (id, username, full_name, password) or None."""
    with _connect() as con:
        row = con.execute(
            "SELECT id, username, full_name, password FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()
        return (row[0], row[1], row[2], row[3]) if row else None

def list_users() -> List[Tuple[int, str, str, str]]:
    with _connect() as con:
        rows = con.execute(
            "SELECT id, username, full_name, created_at FROM users ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

# --- Password helpers ---
def hash_password(plain: str, scheme: str = "sha256") -> str:
    if scheme == "bcrypt":
        import bcrypt  # pip install bcrypt
        hashed = bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt())
        return "bcrypt$" + hashed.decode("utf-8")
    # salted SHA-256
    salt = secrets.token_hex(16)
    hexhash = hashlib.sha256((salt + plain).encode("utf-8")).hexdigest()
    return f"sha256${salt}${hexhash}"

def verify_password(plain: str, stored: str) -> bool:
    if not isinstance(stored, str):
        return False
    try:
        if stored.startswith("bcrypt$"):
            try:
                import bcrypt
            except Exception:
                return False
            hashed = stored[len("bcrypt$"):].encode("utf-8")
            return bcrypt.checkpw(plain.encode("utf-8"), hashed)
        if stored.startswith("sha256$"):
            # sha256$<salt>$<hash>
            try:
                _algo, salt, hexhash = stored.split("$", 2)
            except ValueError:
                return False
            check = hashlib.sha256((salt + plain).encode("utf-8")).hexdigest()
            return hmac.compare_digest(check, hexhash)
        # --- legacy fallback ---
        # 1) legacy unsalted hex SHA-256 of plain
        if len(stored) == 64 and all(c in "0123456789abcdef" for c in stored.lower()):
            return hashlib.sha256(plain.encode("utf-8")).hexdigest() == stored
        # 2) (dev) plain-text stored
        return hmac.compare_digest(plain, stored)
    except Exception:
        return False

# Ensure table on import
ensure_schema()

if __name__ == "__main__":
    # Optional one-time seeding:
    # create_user("admin", "Admin", "admin123", scheme="sha256")  # or "bcrypt" if you have bcrypt installed
    pass
