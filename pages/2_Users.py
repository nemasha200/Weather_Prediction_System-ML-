# pages/2_Users.py — Users admin (list / edit / delete)
import os
import sqlite3
from contextlib import closing
from typing import Optional

import streamlit as st
from db import DB_PATH, hash_password, ensure_schema  # <-- use shared helpers

st.set_page_config(page_title="Users", page_icon="👥", layout="wide")

# ----------------------
# Auth guard (redirect to Login if not logged in)
# ----------------------
if not st.session_state.get("auth_ok", False):
    try:
        st.switch_page("pages/0_Login.py")
    except Exception:
        try:
            st.switch_page("🔐 Login")
        except Exception:
            st.warning("Please log in first (see 🔐 Login page).", icon="🔑")
            st.stop()

# Ensure DB schema exists (safe to call repeatedly)
ensure_schema()

# ----------------------
# DB helpers (use the resolved DB_PATH from db.py)
# ----------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def list_users():
    with closing(get_conn()) as conn, closing(conn.cursor()) as cur:
        # Include password so admin can see what’s stored (hash/format)
        cur.execute("""
            SELECT id, full_name, username, password
            FROM users
            ORDER BY created_at DESC, id DESC
        """)
        return [dict(r) for r in cur.fetchall()]

def delete_user(user_id: int):
    with closing(get_conn()) as conn, closing(conn.cursor()) as cur:
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()

def update_user(user_id: int, full_name: str, username: str, new_password: Optional[str]):
    params = [full_name, username]
    sql = "UPDATE users SET full_name = ?, username = ?"
    if new_password and new_password.strip():
        # ✅ use secure salted hashing from db.py
        hashed = hash_password(new_password.strip())
        sql += ", password = ?"
        params.append(hashed)
    sql += " WHERE id = ?"
    params.append(user_id)
    with closing(get_conn()) as conn, closing(conn.cursor()) as cur:
        cur.execute(sql, params)
        conn.commit()

# ----------------------
# Query param actions (?edit=<id>, ?delete=<id>)
# ----------------------
qp = dict(st.query_params)

if "delete" in qp:
    try:
        val = qp["delete"]
        if isinstance(val, list):
            val = val[0]
        delete_user(int(val))
        st.success("User deleted.")
    except Exception as e:
        st.error(f"Delete failed: {e}")
    qp.pop("delete", None)
    st.query_params = qp
    st.rerun()

_raw_edit = qp.get("edit")
if isinstance(_raw_edit, list):
    _raw_edit = _raw_edit[0]
edit_id = int(_raw_edit) if (_raw_edit and str(_raw_edit).isdigit()) else None

# ----------------------
# Styles
# ----------------------
st.markdown("""
<style>
header[data-testid="stHeader"] { background-color:#003366 !important; }
header[data-testid="stHeader"] * { color:white !important; }

/* Action buttons */
a.btn { display:inline-block; padding:6px 12px; border-radius:8px;
        color:white !important; text-decoration:none !important; font-weight:600;
        border:1px solid rgba(255,255,255,.2); box-shadow:0 1px 4px rgba(0,0,0,.12); }
a.btn.edit { background:#7c3aed; }     /* purple */
a.btn.edit:hover { background:#6d28d9; }
a.btn.del  { background:#f97316; }     /* orange */
a.btn.del:hover  { background:#ea580c; }

/* Table */
table.users { width:100%; border-collapse:collapse; }
table.users th, table.users td { padding:10px 12px; border-bottom:1px solid rgba(0,0,0,.08); }
table.users th { text-align:left; background:rgba(255,255,255,.6); }
tr:hover { background:rgba(0,0,0,.035); }
code.small { font-size:.85em; word-break:break-all; }
</style>
""", unsafe_allow_html=True)

# ----------------------
# Header + back button
# ----------------------
c1, c2 = st.columns([1,1])
with c1:
    st.title("👥 Registered Users")
with c2:
    st.markdown(
        '<div style="text-align:right;margin-top:8px;">'
        '<a class="btn" style="background:#2563eb" href="/">⬅️ Back to Forecast</a>'
        '</div>',
        unsafe_allow_html=True
    )


# ----------------------
# Inline editor
# ----------------------
if edit_id is not None:
    try:
        records = list_users()
        row = next((r for r in records if r["id"] == edit_id), None)
        if not row:
            st.error("User not found.")
        else:
            with st.expander(f"✏️ Edit user #{edit_id}", expanded=True):
                with st.form("edit_user_form", clear_on_submit=False):
                    full_name = st.text_input("Full name", value=row["full_name"])
                    username  = st.text_input("Username", value=row["username"])
                    new_pw    = st.text_input(
                        "New password (leave blank to keep current)",
                        type="password",
                        help="If provided, it will be stored securely (salted)."
                    )
                    submitted = st.form_submit_button("Save changes")
                if submitted:
                    try:
                        update_user(edit_id, full_name, username, new_pw)
                        st.success("User updated.")
                        qp.pop("edit", None)
                        st.query_params = qp
                        st.rerun()
                    except sqlite3.IntegrityError as ie:
                        st.error(f"Username must be unique: {ie}")
                    except Exception as e:
                        st.error(f"Update failed: {e}")
    except Exception as e:
        st.error(f"Could not load user for editing: {e}")

st.markdown("---")

# ----------------------
# Users table
# ----------------------
try:
    users = list_users()
    if not users:
        st.info("No users found.")
    else:
        rows_html = []
        for r in users:
            uid   = r["id"]
            fname = (r["full_name"] or "").replace("<","&lt;")
            uname = (r["username"] or "").replace("<","&lt;")
            pwd   = (r["password"] or "")
            rows_html.append(f"""
            <tr>
              <td>{uid}</td>
              <td>{fname}</td>
              <td>{uname}</td>
              <td><code class="small">{pwd}</code></td>
              <td>
                <a class="btn edit" href="?edit={uid}">Edit</a>
                &nbsp;
                <a class="btn del" href="?delete={uid}" onclick="return confirm('Delete user #{uid}?');">Delete</a>
              </td>
            </tr>
            """)

        st.markdown(
            """
            <table class="users">
              <thead>
                <tr>
                  <th style="width:80px;">ID</th>
                  <th>Full name</th>
                  <th>Username</th>
                  <th>Password (stored)</th>
                  <th style="width:200px;">Action</th>
                </tr>
              </thead>
              <tbody>
            """ + "\n".join(rows_html) + """
              </tbody>
            </table>
            """,
            unsafe_allow_html=True
        )
except Exception as e:
    st.error(f"Failed to load users: {e}")
