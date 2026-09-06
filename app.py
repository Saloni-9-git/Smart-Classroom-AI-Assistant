import os
from datetime import datetime, date
import time
import uuid
import json
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from auth import AuthManager
from rag import SmartRAG
from summarizer import SmartSummarizer
from quiz import QuizGenerator
from attendance import AttendanceSystem
from database import db
from gemini_client import gemini


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Classroom AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL OBJECTS
# ============================================================

auth = AuthManager()
summarizer = SmartSummarizer()
quiz_generator = QuizGenerator()


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "user": None,
    "rag": None,
    "uploaded_files": [],
    "material_uploaded": False,
    "show_reset": False,
    "quiz_data": None,
    "user_answers": {},
    "quiz_submitted": False,
    "last_score": None,
    "last_quiz_topic": "",
    "study_plan": [],
    # Chat history is kept separately for each conversation.
    "chat_sessions": {},
    "active_chat_id": None,
    "current_user_email": None,
    "notes_jobs": [],
    "current_note_job_id": None,
    "history_open_chat_id": None,
    "history_open_note_id": None
}


for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATABASE SUPPORT
# ============================================================

def initialize_extra_tables():
    """
    Creates additional tables required by the new modules.
    Existing tables are not modified.
    """

    try:
        db.cursor.execute("""
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

        db.conn.commit()

    except Exception as e:
        print("Extra table error:", e)


initialize_extra_tables()


# ============================================================
# CUSTOM CSS
# ============================================================

def load_css():

    st.html("""
<style>

/* =========================
   GENERAL
========================= */

.stApp {
    background: #f8fafc;
}

h1, h2, h3 {
    color: #172033;
}

/* =========================
   LOGIN PAGE
========================= */

.brand {
    text-align: center;
    padding: 25px 0 30px 0;
}

.brand-icon {
    font-size: 52px;
}

.brand-title {
    font-size: 42px;
    font-weight: 800;
    color: #172033;
}

.brand-subtitle {
    color: #64748b;
    font-size: 17px;
}

.feature-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 5px 20px rgba(15,23,42,0.05);
}

.feature-title {
    font-size: 19px;
    font-weight: 700;
    color: #172033;
    margin-bottom: 8px;
}

.feature-text {
    color: #64748b;
    line-height: 1.6;
}

.login-card {
    background: white;
    border-radius: 20px;
    padding: 28px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 35px rgba(15,23,42,0.08);
}

/* =========================
   DASHBOARD
========================= */

.dashboard-header {
    background: linear-gradient(
        135deg,
        #111827,
        #334155
    );

    padding: 28px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.dashboard-header h1 {
    color: white;
    margin-bottom: 5px;
}

.dashboard-header p {
    color: #cbd5e1;
}

/* =========================
   SECTION CARD
========================= */

.section-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    margin-bottom: 18px;
}

/* =========================
   HISTORY SIDEBAR
========================= */

.history-sidebar-item {
    font-size: 13px;
    color: #334155;
    margin: 3px 0;
}

.history-sidebar-meta {
    font-size: 11px;
    color: #94a3b8;
}

</style>
""")


load_css()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def initialize_rag():

    if st.session_state.rag is None:
        st.session_state.rag = SmartRAG()




# ============================================================
# PERSISTENT BACKEND STORAGE
# ============================================================
# Streamlit reruns the script whenever the user changes a widget.
# Session state is therefore NOT enough for long-term history.
# These small JSON files keep user data outside session_state.
# For a production cloud deployment, move this storage to a real
# database/object store (e.g. PostgreSQL/Supabase + object storage).

PERSISTENT_ROOT = Path("persistent_data")
PERSISTENT_ROOT.mkdir(parents=True, exist_ok=True)


def current_user_email():
    email = st.session_state.get("current_user_email")
    if email:
        return email.strip().lower()
    user = st.session_state.get("user")
    if user and len(user) > 2:
        return str(user[2]).strip().lower()
    return ""


def user_storage_dir(email=None):
    email = (email or current_user_email()).strip().lower()
    if not email:
        return PERSISTENT_ROOT / "anonymous"
    user_hash = hashlib.sha256(email.encode("utf-8")).hexdigest()[:24]
    path = PERSISTENT_ROOT / "users" / user_hash
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def chat_store_path(email=None):
    return user_storage_dir(email) / "chat_history.json"


def material_store_path(email=None):
    return user_storage_dir(email) / "materials.json"


def notes_store_dir(email=None):
    path = user_storage_dir(email) / "notes_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_chat_history():
    if current_user_email():
        atomic_write_json(
            chat_store_path(),
            st.session_state.get("chat_sessions", {})
        )


def load_material_records(email=None):
    records = read_json(material_store_path(email), [])
    return records if isinstance(records, list) else []


def save_material_records(records, email=None):
    atomic_write_json(material_store_path(email), records)


def load_notes_jobs(email=None):
    result = []
    for path in sorted(notes_store_dir(email).glob("*.json"), reverse=True):
        job = read_json(path, None)
        if isinstance(job, dict):
            result.append(job)
    return result


def note_job_path(job_id, email=None):
    return notes_store_dir(email) / f"{job_id}.json"


def save_note_job(job):
    atomic_write_json(
        note_job_path(job["id"], job.get("user_email")),
        job
    )


def load_note_job(job_id, email=None):
    return read_json(note_job_path(job_id, email), {})


def restore_user_persistent_state():
    """Load chat/material history when the user logs in again."""
    email = current_user_email()
    if not email:
        return

    # IMPORTANT UX RULE:
    # Previous conversations remain permanently stored, but they are NOT
    # loaded into the active workspace after login. The History page reads
    # them directly from persistent storage when the user asks for them.
    st.session_state.chat_sessions = {}
    st.session_state.active_chat_id = None

    records = load_material_records(email)
    st.session_state.uploaded_files = [
        r.get("filename") for r in records
        if isinstance(r, dict) and r.get("filename")
    ]
    st.session_state.material_uploaded = bool(records)
    st.session_state.notes_jobs = [
        j.get("id") for j in load_notes_jobs(email) if j.get("id")
    ]


def ensure_materials_loaded():
    """Rebuild the in-memory RAG index from persisted material files."""
    initialize_rag()
    if getattr(st.session_state.rag, "text_chunks", None):
        return

    records = load_material_records()
    loaded = 0

    for record in records:
        path = record.get("path")
        if not path or not os.path.exists(path):
            continue
        try:
            st.session_state.rag.process_file(path)
            loaded += 1
        except Exception as e:
            print(f"Could not restore material {path}: {e}")

    st.session_state.uploaded_files = [
        r.get("filename") for r in records if r.get("filename")
    ]
    st.session_state.material_uploaded = bool(records and loaded >= 0)


# ============================================================
# BACKGROUND SMART-NOTES WORKER
# ============================================================

@st.cache_resource

def get_note_executor():
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="smart-notes")


@st.cache_resource

def get_active_note_jobs():
    return set()


def run_note_job(job_id, email):
    """Runs without Streamlit UI/session state so navigation cannot cancel it."""
    active_jobs = get_active_note_jobs()
    try:
        job = load_note_job(job_id, email)
        if not job:
            return

        job["status"] = "running"
        job["started_at"] = datetime.utcnow().isoformat()
        save_note_job(job)

        chunks = job.get("chunks", [])
        partial_notes = job.get("partial_notes", [])
        total = len(chunks)

        for index in range(len(partial_notes), total):
            chunk = chunks[index]
            prompt = build_notes_prompt(
                job["note_type"], chunk, index + 1, total
            )
            notes_part = safe_generate(prompt)
            partial_notes.append(notes_part)

            job["partial_notes"] = partial_notes
            job["progress"] = round(len(partial_notes) / max(total, 1), 3)
            job["status"] = "running"
            save_note_job(job)

        notes = combine_note_summaries(
            job["note_type"], partial_notes
        )

        job["notes"] = notes
        job["status"] = "completed"
        job["progress"] = 1.0
        job["completed_at"] = datetime.utcnow().isoformat()
        job["error"] = ""
        # Chunks and partial notes are no longer needed once final notes exist.
        job.pop("chunks", None)
        job.pop("partial_notes", None)
        save_note_job(job)

    except Exception as e:
        job = load_note_job(job_id, email)
        if job:
            job["status"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = datetime.utcnow().isoformat()
            save_note_job(job)
    finally:
        active_jobs.discard(job_id)


def start_note_job(note_type, text, email=None):
    email = (email or current_user_email()).strip().lower()
    chunks = split_text_for_ai(text, max_chars=5000, overlap=250)
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "user_email": email,
        "note_type": note_type,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.utcnow().isoformat(),
        "chunks": chunks,
        "partial_notes": [],
        "notes": "",
        "error": ""
    }
    save_note_job(job)
    get_active_note_jobs().add(job_id)
    get_note_executor().submit(run_note_job, job_id, email)
    return job_id


def resume_note_jobs_for_user(email=None):
    """Recover pending jobs after a normal rerun/process restart."""
    email = (email or current_user_email()).strip().lower()
    if not email:
        return
    active_jobs = get_active_note_jobs()
    for job in load_notes_jobs(email):
        job_id = job.get("id")
        if not job_id or job.get("status") not in {"pending", "running"}:
            continue
        if job_id in active_jobs:
            continue
        active_jobs.add(job_id)
        # A running job without a live worker is safe to resume from its
        # saved partial_notes checkpoint.
        get_note_executor().submit(run_note_job, job_id, email)


def get_latest_note_jobs(email=None):
    jobs = load_notes_jobs(email)
    return jobs[:10]

def initialize_chat_state():
    """Create/load only the current in-session chat.

    Persistent old conversations are deliberately NOT loaded here.
    They are available through the dedicated History page.
    """
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    if not st.session_state.chat_sessions:
        chat_id = str(uuid.uuid4())
        st.session_state.chat_sessions[chat_id] = {
            "title": "New Chat",
            "messages": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        st.session_state.active_chat_id = chat_id

    elif st.session_state.active_chat_id not in st.session_state.chat_sessions:
        st.session_state.active_chat_id = next(
            iter(st.session_state.chat_sessions)
        )


def new_chat():
    """Start a fresh conversation without deleting previous chats."""
    initialize_chat_state()
    chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    st.session_state.active_chat_id = chat_id
    save_chat_history()


def active_chat():
    initialize_chat_state()
    return st.session_state.chat_sessions[
        st.session_state.active_chat_id
    ]


def chat_title_from_question(question):
    title = " ".join(question.strip().split())
    if len(title) > 38:
        title = title[:38].rstrip() + "..."
    return title or "New Chat"


def render_chat_sidebar():
    """Legacy helper retained for compatibility; history now has its own page."""
    return


def build_conversational_query(messages, question):
    """
    Give RAG only a small amount of recent conversation context.
    This helps follow-up questions such as 'explain that again'
    without sending the entire chat history to the model.
    """
    if not messages:
        return question.strip()

    recent = messages[-6:]
    context_parts = []

    for message in recent:
        role = message.get("role", "")
        content = message.get("content", "")
        if not content:
            continue

        # Keep context deliberately small to avoid token-limit problems.
        content = content[:1200]

        if role == "user":
            context_parts.append(f"Student: {content}")
        elif role == "assistant":
            context_parts.append(f"AI: {content}")

    if not context_parts:
        return question.strip()

    return f"""
Use the recent conversation only to understand references such as
'it', 'that', 'this concept', or 'explain again'. Answer the CURRENT
QUESTION using the uploaded learning material.

Recent conversation:
{chr(10).join(context_parts)}

CURRENT QUESTION:
{question.strip()}
""".strip()


def split_text_for_ai(text, max_chars=5500, overlap=300):
    """
    Conservative character-based chunking.
    ~4 characters/token is only an estimate, so we leave plenty of room
    for instructions and the model's answer.
    """
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            # Prefer breaking at a sentence/word boundary.
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end)
            )
            if boundary > start + int(max_chars * 0.60):
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def safe_generate(prompt, retries=2):
    """
    Retry rate-limit/request-size failures with a smaller prompt.
    The primary protection is chunking before this function is called.
    """
    for attempt in range(retries + 1):
        try:
            return gemini.generate(prompt)

        except Exception as e:
            message = str(e).lower()

            rate_problem = (
                "rate_limit" in message
                or "rate limit" in message
                or "413" in message
                or "requested" in message and "limit" in message
                or "tokens per minute" in message
            )

            if attempt >= retries or not rate_problem:
                raise

            # Give the provider's rolling TPM window time to reset.
            wait_seconds = 20 * (attempt + 1)
            time.sleep(wait_seconds)

    raise RuntimeError("AI generation failed.")


def build_notes_prompt(note_type, chunk, chunk_number, total_chunks):
    """Strong, coverage-first prompt for one material chunk."""
    if note_type == "⚡ Quick Revision":
        instructions = """
Create high-quality QUICK REVISION NOTES.
- Capture every important definition, concept, classification, formula,
  process, step, example, fact, date/name/term and exam-worthy point.
- Use compact bullets and clear headings.
- Preserve technical terminology from the material.
- Do not oversummarize away important details.
- Do not invent information.
- Prefer coverage + clarity over decorative wording.
"""
    elif note_type == "📘 Standard Notes":
        instructions = """
Create complete STANDARD CLASSROOM NOTES.
- Cover the material systematically from beginning to end.
- Use logical headings and subheadings.
- Include definitions, explanations, classifications, formulas, steps,
  examples, advantages/disadvantages, comparisons and important facts when present.
- Preserve important terminology and relationships between concepts.
- Do not omit important information merely to make the notes shorter.
- Do not invent information.
"""
    else:
        instructions = """
Create complete BEGINNER-FRIENDLY NOTES for a student who is learning this
for the first time.
- Explain every important concept in simple English.
- Define difficult words before using them.
- Explain processes step-by-step.
- Include formulas and explain what each symbol means when the material does so.
- Include examples, comparisons, memory tips and common exam points when supported.
- Do not assume prior knowledge.
- Do not invent information.
"""

    return f"""
You are an expert educational note-making assistant.

{instructions}

This is section {chunk_number} of {total_chunks}. The final notes will combine
all sections, so preserve information needed for continuity. Do not mention
that you are working on a chunk in the student-facing notes.

SOURCE MATERIAL:
{chunk}
""".strip()


def combine_note_summaries(note_type, summaries):
    """
    Combine already-short chunk summaries. If there are many summaries,
    combine them in batches first so no single request becomes huge.
    """
    if len(summaries) == 1:
        return summaries[0]

    current = summaries[:]

    while len(current) > 4:
        next_round = []

        for i in range(0, len(current), 4):
            batch = current[i:i + 4]
            batch_text = "\n\n--- NEXT SUMMARY ---\n\n".join(batch)

            prompt = f"""
You are organizing study notes.

Combine the following partial {note_type} notes into one concise,
well-structured set of notes.

Rules:
- Preserve important facts from every summary.
- Remove duplication.
- Do not invent information.
- Use clear headings and bullets.
- Keep it concise.

PARTIAL NOTES:
{batch_text}
""".strip()

            next_round.append(safe_generate(prompt))

        current = next_round

    final_text = "\n\n---\n\n".join(current)

    final_prompt = f"""
Create the final {note_type} from the partial notes below.

Rules:
- Preserve the important information.
- Remove repeated points.
- Organize logically.
- Do not introduce facts not supported by these notes.
- Keep the result useful for a student.

PARTIAL NOTES:
{final_text}
""".strip()

    return safe_generate(final_prompt)


def clear_user_session():

    st.session_state.user = None
    st.session_state.rag = None
    st.session_state.uploaded_files = []
    st.session_state.material_uploaded = False
    st.session_state.quiz_data = None
    st.session_state.user_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.last_score = None
    st.session_state.study_plan = []
    st.session_state.chat_sessions = {}
    st.session_state.active_chat_id = None
    st.session_state.current_user_email = None
    st.session_state.notes_jobs = []

    st.rerun()


def safe_filename(filename):

    filename = os.path.basename(filename)

    return filename.replace("/", "_").replace("\\", "_")


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    # Hide sidebar on login
    st.html("""
<style>
[data-testid="stSidebar"] {
    display: none;
}
</style>
""")

    # -------------------------
    # BRAND
    # -------------------------

    st.html("""
<div class="brand">

    <div class="brand-icon">
        🎓
    </div>

    <div class="brand-title">
        Smart Classroom AI
    </div>

    <div class="brand-subtitle">
        Your Intelligent Learning Companion
    </div>

</div>
""")

    left, right = st.columns(
        [1.1, 1],
        gap="large"
    )

    # ========================================================
    # LEFT SIDE
    # ========================================================

    with left:

        st.html("""
<div class="feature-card">

    <div class="feature-title">
        🚀 Learn Smarter with AI
    </div>

    <div class="feature-text">
        Ask questions, understand difficult concepts
        and get intelligent assistance for your studies.
    </div>

</div>
""")

        st.html("""
<div class="feature-card">

    <div class="feature-title">
        📚 Smart Learning Materials
    </div>

    <div class="feature-text">
        Upload PDF, Word, Excel and image-based
        learning materials and interact with your
        study content.
    </div>

</div>
""")

        st.html("""
<div class="feature-card">

    <div class="feature-title">
        🤖 AI-Powered Learning
    </div>

    <div class="feature-text">
        Generate smart notes, solve doubts, create
        quizzes and analyze your learning progress
        from one platform.
    </div>

</div>
""")

        st.html("""
<div class="feature-card">

    <div class="feature-title">
        📊 Personalized Progress
    </div>

    <div class="feature-text">
        Track quiz performance, attendance and study
        activities to understand your learning progress.
    </div>

</div>
""")

    # ========================================================
    # RIGHT SIDE
    # ========================================================

    with right:

        st.html("""
<div class="login-card">
    <h2>Welcome Back 👋</h2>
    <p style="color:#64748b;">
        Sign in to continue to your classroom dashboard.
    </p>
</div>
""")

        login_tab, signup_tab = st.tabs(
            [
                "🔐 Login",
                "✨ Create Account"
            ]
        )

        # ====================================================
        # LOGIN
        # ====================================================

        with login_tab:

            email = st.text_input(
                "Email Address",
                placeholder="Enter your email",
                key="login_email"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password"
            )

            login_col, forgot_col = st.columns(2)

            with login_col:

                if st.button(
                    "🔐 Login",
                    use_container_width=True,
                    key="login_button"
                ):

                    if not email.strip():

                        st.warning(
                            "Please enter your email."
                        )

                    elif not password:

                        st.warning(
                            "Please enter your password."
                        )

                    else:

                        user = auth.login(
                            email.strip(),
                            password
                        )

                        if user:

                            st.session_state.user = user
                            st.session_state.current_user_email = str(user[2]).strip().lower()
                            st.session_state.rag = SmartRAG()
                            restore_user_persistent_state()
                            resume_note_jobs_for_user()

                            st.success(
                                "Login successful!"
                            )

                            st.rerun()

                        else:

                            st.error(
                                "Invalid email or password."
                            )

            with forgot_col:

                if st.button(
                    "🔑 Forgot Password?",
                    use_container_width=True,
                    key="forgot_password_button"
                ):

                    st.session_state.show_reset = True

            # =================================================
            # RESET PASSWORD
            # =================================================

            if st.session_state.show_reset:

                st.divider()

                st.subheader("🔑 Reset Password")

                reset_email = st.text_input(
                    "Registered Email",
                    placeholder="Enter registered email",
                    key="reset_email"
                )

                new_password = st.text_input(
                    "New Password",
                    type="password",
                    placeholder="Enter new password",
                    key="reset_new_password"
                )

                confirm_password = st.text_input(
                    "Confirm New Password",
                    type="password",
                    placeholder="Confirm password",
                    key="reset_confirm_password"
                )

                reset_col, cancel_col = st.columns(2)

                with reset_col:

                    if st.button(
                        "Reset Password",
                        use_container_width=True,
                        key="reset_password_button"
                    ):

                        if not reset_email.strip():

                            st.warning(
                                "Enter your registered email."
                            )

                        elif len(new_password) < 6:

                            st.warning(
                                "Password must contain at least 6 characters."
                            )

                        elif new_password != confirm_password:

                            st.error(
                                "Passwords do not match."
                            )

                        else:

                            try:

                                # If reset_password exists
                                if hasattr(
                                    auth,
                                    "reset_password"
                                ):

                                    success = auth.reset_password(
                                        reset_email.strip(),
                                        new_password
                                    )

                                else:

                                    # Fallback for current auth.py
                                    # -----------------------------

                                    hashed = auth.hash_password(
                                        new_password
                                    )

                                    db.cursor.execute(
                                        """
                                        UPDATE users
                                        SET password=?
                                        WHERE email=?
                                        """,
                                        (
                                            hashed,
                                            reset_email.strip()
                                        )
                                    )

                                    db.conn.commit()

                                    success = (
                                        db.cursor.rowcount > 0
                                    )

                                if success:

                                    st.success(
                                        "Password reset successfully. "
                                        "You can now login."
                                    )

                                    st.session_state.show_reset = False

                                else:

                                    st.error(
                                        "Email not found."
                                    )

                            except Exception as e:

                                st.error(
                                    f"Password reset failed: {e}"
                                )

                with cancel_col:

                    if st.button(
                        "Cancel",
                        use_container_width=True,
                        key="cancel_reset_button"
                    ):

                        st.session_state.show_reset = False
                        st.rerun()

        # ====================================================
        # SIGNUP
        # ====================================================

        with signup_tab:

            name = st.text_input(
                "Full Name",
                placeholder="Enter your name",
                key="signup_name"
            )

            signup_email = st.text_input(
                "Email Address",
                placeholder="Enter your email",
                key="signup_email"
            )

            signup_password = st.text_input(
                "Password",
                type="password",
                placeholder="Create a password",
                key="signup_password"
            )

            role = st.selectbox(
                "Account Type",
                [
                    "Student",
                    "Teacher"
                ],
                key="signup_role"
            )

            if st.button(
                "✨ Create Account",
                use_container_width=True,
                key="signup_button"
            ):

                if not name.strip():

                    st.warning(
                        "Please enter your name."
                    )

                elif not signup_email.strip():

                    st.warning(
                        "Please enter your email."
                    )

                elif len(signup_password) < 6:

                    st.warning(
                        "Password must contain at least 6 characters."
                    )

                else:

                    success = auth.signup(
                        name.strip(),
                        signup_email.strip(),
                        signup_password,
                        role
                    )

                    if success:

                        st.success(
                            "Account created successfully! "
                            "Please login."
                        )

                    else:

                        st.error(
                            "Unable to create account. "
                            "Email may already exist."
                        )

    st.html("""
<div style="
    text-align:center;
    color:#94a3b8;
    margin-top:30px;
    padding-bottom:20px;
">
    Smart Classroom AI • Intelligent Education Platform
</div>
""")


# ============================================================
# LEARNING MATERIAL
# ============================================================

def upload_learning_material():

    ensure_materials_loaded()
    initialize_chat_state()

    st.title("📚 Learning Material")

    st.caption(
        "Upload your study materials and ask questions directly "
        "from the same page."
    )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    uploaded_files = st.file_uploader(
        "Upload PDF, Word, Excel or Images",
        type=[
            "pdf",
            "docx",
            "xlsx",
            "xls",
            "png",
            "jpg",
            "jpeg"
        ],
        accept_multiple_files=True,
        key="learning_material_uploader"
    )

    if uploaded_files:

        os.makedirs("uploads", exist_ok=True)

        processed_count = 0

        for uploaded_file in uploaded_files:

            clean_name = safe_filename(uploaded_file.name)

            # Store each user's material in a persistent user folder.
            material_dir = user_storage_dir() / "materials"
            material_dir.mkdir(parents=True, exist_ok=True)
            save_path = material_dir / clean_name

            # Prevent duplicate processing when the same file is uploaded again.
            existing_records = load_material_records()
            existing_names = {r.get("filename") for r in existing_records}
            if clean_name in existing_names:
                continue

            try:

                with open(save_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                with st.spinner(f"Processing {clean_name}..."):
                    st.session_state.rag.process_file(str(save_path))

                existing_records.append({
                    "filename": clean_name,
                    "path": str(save_path),
                    "uploaded_at": datetime.utcnow().isoformat()
                })
                save_material_records(existing_records)

                st.session_state.uploaded_files.append(clean_name)
                processed_count += 1

            except Exception as e:

                st.error(
                    f"Could not process {clean_name}: {e}"
                )

        if processed_count > 0:

            st.session_state.material_uploaded = True

            st.success(
                f"{processed_count} file(s) processed successfully."
            )

    # ========================================================
    # UPLOADED FILES
    # ========================================================

    if st.session_state.uploaded_files:

        st.divider()
        st.subheader("📄 Uploaded Files")

        for index, filename in enumerate(
            st.session_state.uploaded_files,
            start=1
        ):
            st.write(f"**{index}.** {filename}")

        st.info(
            f"Total files: {len(st.session_state.uploaded_files)}"
        )

    # ========================================================
    # CHATBOT
    # ========================================================

    st.divider()

    st.subheader("🤖 AI Classroom Chatbot")

    current = active_chat()

    if not current["messages"]:
        st.info(
            "Start a conversation below. Your previous chats will stay "
            "saved when you click **＋ New Chat**."
        )

    # Render complete conversation every rerun.
    for message in current["messages"]:

        role = message.get("role", "assistant")

        with st.chat_message(
            "user" if role == "user" else "assistant"
        ):

            st.markdown(message.get("content", ""))

            if role == "assistant":

                sources = message.get("sources", [])

                st.markdown("**📖 Source References**")

                if sources:
                    for source in sources:
                        st.info(source)
                else:
                    st.caption(
                        "No specific source chunk was returned."
                    )

    question = st.chat_input(
        "Ask another question about your uploaded material...",
        key="material_chat_input"
    )

    if question:

        if not st.session_state.material_uploaded:

            st.warning(
                "Please upload learning material first."
            )

            return

        question = question.strip()

        if not question:
            return

        # Save the question before generating the answer.
        now_iso = datetime.utcnow().isoformat()

        current["messages"].append({
            "role": "user",
            "content": question,
            "created_at": now_iso
        })
        current["updated_at"] = now_iso

        # The first question becomes the chat title.
        if current["title"] == "New Chat":
            current["title"] = chat_title_from_question(question)

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "AI is analyzing your material..."
            ):

                try:

                    query = build_conversational_query(
                        current["messages"][:-1],
                        question
                    )

                    result = st.session_state.rag.ask(query)

                    answer = result.get(
                        "answer",
                        "No answer generated."
                    )

                    sources = result.get("sources", [])

                    st.markdown(answer)

                    st.markdown("**📖 Source References**")

                    if sources:
                        for source in sources:
                            st.info(source)
                    else:
                        st.caption(
                            "No specific source chunk was returned."
                        )

                    # Store answer permanently in this chat.
                    current["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "created_at": datetime.utcnow().isoformat()
                    })
                    current["updated_at"] = datetime.utcnow().isoformat()
                    save_chat_history()

                except Exception as e:

                    error_message = f"Chatbot error: {e}"

                    st.error(error_message)

                    current["messages"].append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "created_at": datetime.utcnow().isoformat()
                    })
                    current["updated_at"] = datetime.utcnow().isoformat()
                    save_chat_history()

# ============================================================
# SMART NOTES
# ============================================================

def notes_summarizer():

    ensure_materials_loaded()

    st.title("📝 AI Smart Notes Generator")
    st.caption(
        "Generate structured notes in the background. "
        "Your previous notes stay safely saved in History."
    )

    if not st.session_state.material_uploaded:
        st.warning("Please upload learning material first.")
        return

    st.write("Choose the level of notes according to your learning style.")

    note_type = st.radio(
        "Notes Level",
        [
            "⚡ Quick Revision",
            "📘 Standard Notes",
            "🌱 Beginner Friendly"
        ],
        horizontal=True,
        key="notes_level"
    )

    resume_note_jobs_for_user()

    if st.button(
        "✨ Generate Smart Notes",
        use_container_width=True,
        key="generate_notes_button"
    ):
        text = " ".join(st.session_state.rag.text_chunks)

        if not text.strip():
            st.warning("No readable text was found in the uploaded material.")
            return

        job_id = start_note_job(note_type, text)
        st.session_state.notes_jobs.append(job_id)
        st.session_state.current_note_job_id = job_id

        st.success(
            "Notes generation started in the background. "
            "You can safely explore other tabs."
        )
        st.rerun()

    # Only show the job the user is currently working on.
    # Older completed notes are intentionally kept out of this page.
    current_id = st.session_state.get("current_note_job_id")

    if current_id:
        job = load_note_job(current_id)

        if job:
            st.divider()
            st.subheader("Current Notes Job")

            status = job.get("status", "unknown")
            title = job.get("note_type", "Smart Notes")
            created = job.get("created_at", "")[:19].replace("T", " ")

            st.markdown(f"### {title}")
            if created:
                st.caption(f"Created: {created}")

            if status in {"pending", "running"}:
                progress = float(job.get("progress", 0) or 0)
                st.progress(
                    progress,
                    text=f"Generating in background... {progress * 100:.0f}%"
                )
                st.info(
                    "You can leave this page. The job continues in the background."
                )

            elif status == "completed":
                st.success("Smart Notes are ready and saved.")
                notes = job.get("notes", "")
                st.markdown(notes)
                st.download_button(
                    "📥 Download Smart Notes",
                    data=notes,
                    file_name="Smart_Notes.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"download_notes_{job.get('id')}"
                )

            elif status == "failed":
                st.error("This notes job failed.")
                if job.get("error"):
                    st.caption(job["error"])

    st.divider()
    st.caption("Previous notes are available from 🕘 History.")

# ============================================================
# QUIZ
# ============================================================

def quiz_section():

    st.title("🎯 AI Quiz Generator")

    st.caption(
        "Generate quizzes and track your performance automatically."
    )

    topic = st.text_input(
        "Enter Topic",
        placeholder="Example: Machine Learning",
        key="quiz_topic"
    )

    difficulty = st.selectbox(
        "Difficulty",
        [
            "Easy",
            "Medium",
            "Hard"
        ],
        key="quiz_difficulty"
    )

    if st.button(
        "🎯 Generate Quiz",
        use_container_width=True,
        key="generate_quiz_button"
    ):

        if not topic.strip():

            st.warning(
                "Please enter a topic."
            )

        else:

            with st.spinner(
                "Generating quiz..."
            ):

                try:

                    quiz_data = quiz_generator.generate_quiz(
                        topic.strip(),
                        difficulty
                    )

                    if not quiz_data:

                        st.error(
                            "Quiz generation failed."
                        )

                    else:

                        st.session_state.quiz_data = quiz_data
                        st.session_state.user_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.last_score = None
                        st.session_state.last_quiz_topic = topic.strip()

                        st.success(
                            "Quiz generated successfully!"
                        )

                except Exception as e:

                    st.error(
                        f"Quiz generation error: {e}"
                    )

    # ========================================================
    # DISPLAY QUIZ
    # ========================================================

    if st.session_state.quiz_data:

        st.divider()

        st.subheader(
            f"📝 Quiz: {st.session_state.last_quiz_topic}"
        )

        for i, question in enumerate(
            st.session_state.quiz_data
        ):

            st.markdown(
                f"### Q{i + 1}. {question['question']}"
            )

            options = question.get(
                "options",
                {}
            )

            answer = st.radio(
                "Choose your answer",
                list(options.keys()),
                format_func=lambda x:
                    f"{x}. {options[x]}",
                key=f"quiz_question_{i}"
            )

            st.session_state.user_answers[i] = answer

        if st.button(
            "✅ Submit Quiz",
            use_container_width=True,
            key="submit_quiz_button"
        ):

            score = 0

            weak_topics = []

            for i, question in enumerate(
                st.session_state.quiz_data
            ):

                selected = st.session_state.user_answers.get(
                    i
                )

                correct = question.get(
                    "correct_answer"
                )

                if selected == correct:

                    score += 1

                else:

                    weak_topics.append(
                        st.session_state.last_quiz_topic
                    )

            total = len(
                st.session_state.quiz_data
            )

            percentage = (
                (score / total) * 100
                if total > 0
                else 0
            )

            st.session_state.last_score = score
            st.session_state.quiz_submitted = True

            # =================================================
            # SAVE SCORE
            # =================================================

            try:

                student_name = st.session_state.user[1]

                weak_topic = (
                    ", ".join(
                        set(weak_topics)
                    )
                    if weak_topics
                    else ""
                )

                db.cursor.execute(
                    """
                    INSERT INTO quiz_scores
                    (
                        student,
                        subject,
                        score,
                        weak_topic
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        student_name,
                        st.session_state.last_quiz_topic,
                        score,
                        weak_topic
                    )
                )

                db.conn.commit()

            except Exception as e:

                st.warning(
                    f"Score could not be saved: {e}"
                )

            # =================================================
            # RESULT
            # =================================================

            st.divider()

            if percentage >= 80:

                st.success(
                    f"🎉 Excellent! Your score is "
                    f"{score}/{total} ({percentage:.0f}%)."
                )

            elif percentage >= 50:

                st.info(
                    f"👍 Good effort! Your score is "
                    f"{score}/{total} ({percentage:.0f}%)."
                )

            else:

                st.warning(
                    f"📚 You scored "
                    f"{score}/{total} ({percentage:.0f}%). "
                    f"Consider revising this topic."
                )

            st.subheader(
                "Correct Answers"
            )

            for i, question in enumerate(
                st.session_state.quiz_data
            ):

                st.write(
                    f"Q{i + 1}: "
                    f"**{question.get('correct_answer', 'N/A')}**"
                )


# ============================================================
# ATTENDANCE
# ============================================================

def attendance_section():

    st.title("📸 Smart Attendance System")

    st.caption(
        "Register student faces and mark attendance using the "
        "existing face recognition module."
    )

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student name",
        key="attendance_student_name"
    )

    register_col, mark_col = st.columns(2)

    # ========================================================
    # REGISTER
    # ========================================================

    with register_col:

        st.markdown(
            "### 👤 Register Student"
        )

        if st.button(
            "📷 Register Face",
            use_container_width=True,
            key="register_face_button"
        ):

            if not student_name.strip():

                st.warning(
                    "Please enter student name."
                )

            else:

                try:

                    attendance = AttendanceSystem()

                    with st.spinner(
                        "Capturing face..."
                    ):

                        attendance.register_student(
                            student_name.strip()
                        )

                    attendance.train_model()

                    st.success(
                        f"{student_name} registered successfully."
                    )

                except Exception as e:

                    st.error(
                        f"Face registration error: {e}"
                    )

    # ========================================================
    # MARK ATTENDANCE
    # ========================================================

    with mark_col:

        st.markdown(
            "### ✅ Mark Attendance"
        )

        if st.button(
            "📸 Start Face Recognition",
            use_container_width=True,
            key="mark_attendance_button"
        ):

            try:

                attendance = AttendanceSystem()

                with st.spinner(
                    "Opening camera..."
                ):

                    label_map = (
                        attendance.train_model()
                    )

                    name = (
                        attendance.mark_attendance(
                            label_map
                        )
                    )

                if name:

                    st.success(
                        f"Attendance marked for {name}."
                    )

                else:

                    st.error(
                        "Face not recognized."
                    )

            except Exception as e:

                st.error(
                    f"Attendance error: {e}"
                )


# ============================================================
# ANALYTICS
# ============================================================

def analytics_section():

    st.title("📊 Student Analytics")

    student_name = st.session_state.user[1]

    st.caption(
        f"Learning analytics for {student_name}"
    )

    # ========================================================
    # QUIZ ANALYTICS
    # ========================================================

    try:

        db.cursor.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(score), 0),
                COALESCE(MAX(score), 0)
            FROM quiz_scores
            WHERE student=?
            """,
            (student_name,)
        )

        quiz_stats = db.cursor.fetchone()

        total_quizzes = quiz_stats[0] or 0
        total_score = quiz_stats[1] or 0
        highest_score = quiz_stats[2] or 0

    except Exception:

        total_quizzes = 0
        total_score = 0
        highest_score = 0

    # ========================================================
    # ATTENDANCE ANALYTICS
    # ========================================================

    try:

        db.cursor.execute(
            """
            SELECT COUNT(*)
            FROM attendance
            WHERE student_name=?
            AND status='Present'
            """,
            (student_name,)
        )

        present_count = (
            db.cursor.fetchone()[0] or 0
        )

        db.cursor.execute(
            """
            SELECT COUNT(*)
            FROM attendance
            WHERE student_name=?
            """,
            (student_name,)
        )

        total_attendance = (
            db.cursor.fetchone()[0] or 0
        )

    except Exception:

        present_count = 0
        total_attendance = 0

    attendance_percentage = (
        (present_count / total_attendance) * 100
        if total_attendance > 0
        else 0
    )

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📝 Quizzes",
            total_quizzes
        )

    with col2:

        st.metric(
            "🏆 Highest Score",
            highest_score
        )

    with col3:

        st.metric(
            "📚 Materials",
            len(
                st.session_state.uploaded_files
            )
        )

    with col4:

        st.metric(
            "📅 Attendance",
            f"{attendance_percentage:.0f}%"
        )

    # ========================================================
    # QUIZ HISTORY
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Quiz Performance History"
    )

    try:

        db.cursor.execute(
            """
            SELECT
                subject,
                score,
                weak_topic,
                date
            FROM quiz_scores
            WHERE student=?
            ORDER BY date DESC
            """,
            (student_name,)
        )

        records = db.cursor.fetchall()

        if records:

            for record in records:

                subject, score, weak_topic, quiz_date = record

                st.write(
                    f"**{subject}** — "
                    f"Score: **{score}** — "
                    f"{quiz_date}"
                )

                if weak_topic:

                    st.caption(
                        f"Weak topic: {weak_topic}"
                    )

                st.divider()

        else:

            st.info(
                "No quiz attempts recorded yet."
            )

    except Exception as e:

        st.error(
            f"Could not load analytics: {e}"
        )

    # ========================================================
    # LEARNING INSIGHT
    # ========================================================

    st.subheader(
        "💡 Learning Insight"
    )

    if total_quizzes == 0:

        st.info(
            "Complete your first quiz to receive "
            "personalized learning insights."
        )

    elif highest_score >= 4:

        st.success(
            "Excellent performance! Keep practicing "
            "to maintain your progress."
        )

    else:

        st.warning(
            "Your performance can improve with regular "
            "revision and practice quizzes."
        )


# ============================================================
# STUDY PLANNER
# ============================================================

def study_planner():

    st.title("📅 Study Planner")

    st.caption(
        "Create a simple personalized study schedule."
    )

    student_name = st.session_state.user[1]

    # ========================================================
    # ADD TASK
    # ========================================================

    with st.form(
        "study_plan_form",
        clear_on_submit=True
    ):

        task = st.text_input(
            "Study Task",
            placeholder="Example: Revise Machine Learning"
        )

        subject = st.text_input(
            "Subject",
            placeholder="Example: Artificial Intelligence"
        )

        plan_date = st.date_input(
            "Study Date",
            value=date.today()
        )

        duration = st.number_input(
            "Duration (minutes)",
            min_value=15,
            max_value=600,
            value=60,
            step=15
        )

        submitted = st.form_submit_button(
            "➕ Add Study Task",
            use_container_width=True
        )

    if submitted:

        if not task.strip():

            st.warning(
                "Please enter a study task."
            )

        else:

            try:

                db.cursor.execute(
                    """
                    INSERT INTO study_plans
                    (
                        student,
                        task,
                        subject,
                        plan_date,
                        duration
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        student_name,
                        task.strip(),
                        subject.strip(),
                        str(plan_date),
                        duration
                    )
                )

                db.conn.commit()

                st.success(
                    "Study task added successfully!"
                )

            except Exception as e:

                st.error(
                    f"Could not add study task: {e}"
                )

    # ========================================================
    # DISPLAY PLAN
    # ========================================================

    st.divider()

    st.subheader(
        "📋 My Study Plan"
    )

    try:

        db.cursor.execute(
            """
            SELECT
                id,
                task,
                subject,
                plan_date,
                duration,
                status
            FROM study_plans
            WHERE student=?
            ORDER BY plan_date ASC
            """,
            (student_name,)
        )

        plans = db.cursor.fetchall()

        if not plans:

            st.info(
                "No study tasks added yet."
            )

        else:

            for plan in plans:

                (
                    plan_id,
                    task_name,
                    subject_name,
                    plan_date_value,
                    minutes,
                    status
                ) = plan

                with st.container(border=True):

                    col1, col2, col3 = st.columns(
                        [2.5, 1.5, 1]
                    )

                    with col1:

                        st.markdown(
                            f"### 📚 {task_name}"
                        )

                        if subject_name:

                            st.caption(
                                f"Subject: {subject_name}"
                            )

                    with col2:

                        st.write(
                            f"📅 {plan_date_value}"
                        )

                        st.write(
                            f"⏱️ {minutes} minutes"
                        )

                    with col3:

                        if status == "Completed":

                            st.success(
                                "Completed"
                            )

                        else:

                            if st.button(
                                "✓ Complete",
                                key=f"complete_plan_{plan_id}"
                            ):

                                db.cursor.execute(
                                    """
                                    UPDATE study_plans
                                    SET status='Completed'
                                    WHERE id=?
                                    """,
                                    (plan_id,)
                                )

                                db.conn.commit()

                                st.rerun()

    except Exception as e:

        st.error(
            f"Could not load study plan: {e}"
        )


# ============================================================
# HISTORY
# ============================================================

def _history_date(value):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime(
            "%d %b %Y • %I:%M %p"
        )
    except Exception:
        return value[:19].replace("T", " ")


def _chat_preview(chat):
    messages = chat.get("messages", [])
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            return " ".join(message["content"].split())[:150]
    return "Empty conversation"


def _chat_last_date(chat):
    messages = chat.get("messages", [])
    for message in reversed(messages):
        if message.get("created_at"):
            return message["created_at"]
    return chat.get("updated_at", "")


def _load_history_chats(email=None):
    data = read_json(chat_store_path(email), {})
    return data if isinstance(data, dict) else {}


def _save_history_chats(chats, email=None):
    atomic_write_json(chat_store_path(email), chats)


def _delete_history_chat(chat_id, email=None):
    chats = _load_history_chats(email)
    if chat_id in chats:
        del chats[chat_id]
        _save_history_chats(chats, email)


def _open_history_chat(chat_id, email=None):
    chats = _load_history_chats(email)
    chat = chats.get(chat_id)

    if not chat:
        return False

    st.session_state.chat_sessions = {chat_id: chat}
    st.session_state.active_chat_id = chat_id
    st.session_state.history_open_chat_id = chat_id
    return True


def _delete_history_note(job_id, email=None):
    path = note_job_path(job_id, email)
    try:
        if path.exists():
            path.unlink()
        return True
    except OSError:
        return False


def history_section():

    email = current_user_email()

    st.html("""
    <div class="history-hero">
        <h1>🕘 History</h1>
        <p>
            Your previous chats, notes and quiz activity are saved here.
            Nothing old is shown on your main workspace unless you open it.
        </p>
    </div>
    """)

    history_type = st.radio(
        "History Type",
        [
            "💬 Questions & Chats",
            "📝 Smart Notes",
            "🎯 Quiz Attempts"
        ],
        horizontal=True,
        key="history_type"
    )

    search = st.text_input(
        "🔍 Search your history",
        placeholder="Search questions, topics, notes or quiz subjects...",
        key="history_search"
    ).strip().lower()

    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if history_type == "💬 Questions & Chats":

        chats = _load_history_chats(email)
        items = []

        for chat_id, chat in chats.items():
            preview = _chat_preview(chat)
            title = chat.get("title", "New Chat")
            haystack = f"{title} {preview}".lower()

            if search and search not in haystack:
                # Also search every user question in the conversation.
                questions = " ".join(
                    m.get("content", "")
                    for m in chat.get("messages", [])
                    if m.get("role") == "user"
                ).lower()
                if search not in questions:
                    continue

            items.append((chat_id, chat))

        items.sort(
            key=lambda item: _chat_last_date(item[1]),
            reverse=True
        )

        if not items:
            st.html("""
            <div class="history-empty">
                💬<br><br>
                No saved conversations match your search.
            </div>
            """)
            return

        st.caption(f"{len(items)} saved conversation(s)")

        for chat_id, chat in items:

            title = chat.get("title", "New Chat")
            preview = _chat_preview(chat)
            messages_count = len(chat.get("messages", []))
            date_text = _history_date(_chat_last_date(chat))

            with st.container(border=True):
                st.markdown(f"**💬 {title}**")
                st.caption(
                    f"{date_text}  •  {messages_count} messages"
                )
                st.write(preview)

                open_col, delete_col = st.columns([5, 1])

                with open_col:
                    if st.button(
                        "Open Conversation",
                        key=f"history_open_chat_{chat_id}",
                        use_container_width=True
                    ):
                        if _open_history_chat(chat_id, email):
                            st.session_state.main_navigation = "📚 Learning Material"
                            st.rerun()

                with delete_col:
                    if st.button(
                        "🗑️",
                        key=f"history_delete_chat_{chat_id}",
                        use_container_width=True
                    ):
                        _delete_history_chat(chat_id, email)
                        st.rerun()

    # ========================================================
    # NOTES HISTORY
    # ========================================================

    elif history_type == "📝 Smart Notes":

        jobs = load_notes_jobs(email)

        filtered = []
        for job in jobs:
            haystack = " ".join([
                job.get("note_type", ""),
                job.get("notes", ""),
                job.get("created_at", "")
            ]).lower()

            if not search or search in haystack:
                filtered.append(job)

        if not filtered:
            st.html("""
            <div class="history-empty">
                📝<br><br>
                No saved Smart Notes match your search.
            </div>
            """)
            return

        st.caption(f"{len(filtered)} saved note job(s)")

        for job in filtered:

            job_id = job.get("id")
            title = job.get("note_type", "Smart Notes")
            status = job.get("status", "unknown")
            created = _history_date(job.get("created_at", ""))
            notes = job.get("notes", "")

            with st.container(border=True):

                st.markdown(f"**📝 {title}**")
                st.caption(f"{created}  •  Status: {status}")

                if status == "completed":
                    preview = " ".join(notes.split())[:180]
                    st.write(preview or "Saved notes")

                    open_col, delete_col = st.columns([5, 1])

                    with open_col:
                        if st.button(
                            "Open Notes",
                            key=f"history_open_note_{job_id}",
                            use_container_width=True
                        ):
                            st.session_state.history_open_note_id = job_id
                            st.rerun()

                    with delete_col:
                        if st.button(
                            "🗑️",
                            key=f"history_delete_note_{job_id}",
                            use_container_width=True
                        ):
                            _delete_history_note(job_id, email)
                            st.rerun()

                elif status in {"pending", "running"}:
                    progress = float(job.get("progress", 0) or 0)
                    st.progress(
                        progress,
                        text=f"Background generation: {progress * 100:.0f}%"
                    )
                    st.caption(
                        "This job is still running and remains saved."
                    )

                else:
                    st.error(job.get("error", "Notes generation failed."))

        # Open one note only when explicitly requested.
        open_note_id = st.session_state.get("history_open_note_id")
        if open_note_id:
            selected = load_note_job(open_note_id, email)
            if selected:
                st.divider()
                st.subheader(f"📖 {selected.get('note_type', 'Smart Notes')}")
                st.caption(
                    _history_date(selected.get("created_at", ""))
                )
                st.markdown(selected.get("notes", ""))

                st.download_button(
                    "📥 Download Smart Notes",
                    data=selected.get("notes", ""),
                    file_name="Smart_Notes.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"history_download_note_{open_note_id}"
                )

                if st.button(
                    "Close Opened Notes",
                    key="close_history_note"
                ):
                    st.session_state.history_open_note_id = None
                    st.rerun()

    # ========================================================
    # QUIZ HISTORY
    # ========================================================

    else:

        student_name = st.session_state.user[1]

        try:
            db.cursor.execute(
                """
                SELECT subject, score, weak_topic, date
                FROM quiz_scores
                WHERE student=?
                ORDER BY date DESC
                """,
                (student_name,)
            )
            records = db.cursor.fetchall()
        except Exception:
            records = []

        if search:
            records = [
                r for r in records
                if search in str(r[0]).lower()
                or search in str(r[2] or "").lower()
            ]

        if not records:
            st.html("""
            <div class="history-empty">
                🎯<br><br>
                No quiz attempts match your search.
            </div>
            """)
            return

        st.caption(f"{len(records)} quiz attempt(s)")

        for subject, score, weak_topic, quiz_date in records:
            with st.container(border=True):
                st.markdown(f"**🎯 {subject}**")
                st.caption(f"Attempted: {quiz_date}")
                st.write(f"Score: **{score}**")
                if weak_topic:
                    st.caption(f"Weak topic: {weak_topic}")


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():

    initialize_rag()

    user = st.session_state.user

    user_name = user[1]
    role = user[4]

    if not st.session_state.get("current_user_email") and len(user) > 2:
        st.session_state.current_user_email = str(user[2]).strip().lower()
        restore_user_persistent_state()

    resume_note_jobs_for_user()

    # ========================================================
    # SIDEBAR
    # ========================================================

    initialize_chat_state()

    with st.sidebar:

        st.title("🎓 Smart Classroom AI")

        st.caption(f"Welcome, {user_name}")

        st.divider()

        choice = st.radio(
            "Navigate",
            [
                "🏠 Dashboard",
                "📚 Learning Material",
                "📝 Smart Notes",
                "🎯 Quiz Generator",
                "📸 Attendance",
                "📊 Analytics",
                "📅 Study Planner"
            ],
            key="main_navigation"
        )

        st.divider()

        st.caption(f"Role: {role}")

        # ====================================================
        # COLLAPSIBLE HISTORY
        # ====================================================
        # Nothing from previous sessions is shown until the
        # user explicitly clicks the arrow.
        with st.sidebar.expander("🕘 History", expanded=False):

            history_tab_chat, history_tab_notes = st.tabs(
                ["💬 Chats", "📝 Notes"]
            )

            # ------------------------------
            # PREVIOUS QUESTIONS / CHATS
            # ------------------------------
            with history_tab_chat:

                saved_chats = _load_history_chats(
                    current_user_email()
                )

                chat_items = list(saved_chats.items())

                chat_items.sort(
                    key=lambda item: _chat_last_date(item[1]),
                    reverse=True
                )

                if not chat_items:
                    st.caption("No previous questions yet.")

                else:
                    for chat_id, chat in chat_items[:20]:

                        title = chat.get(
                            "title",
                            "New Chat"
                        )

                        preview = _chat_preview(chat)

                        # The button is intentionally compact so the
                        # sidebar stays clean even with many chats.
                        if st.button(
                            f"💬 {title}",
                            key=f"sidebar_history_chat_{chat_id}",
                            use_container_width=True
                        ):
                            if _open_history_chat(
                                chat_id,
                                current_user_email()
                            ):
                                st.session_state.main_navigation = (
                                    "📚 Learning Material"
                                )
                                st.rerun()

                        st.caption(
                            preview[:75]
                            + ("..." if len(preview) > 75 else "")
                        )

            # ------------------------------
            # PREVIOUS SMART NOTES
            # ------------------------------
            with history_tab_notes:

                saved_notes = load_notes_jobs(
                    current_user_email()
                )

                if not saved_notes:
                    st.caption("No previous notes yet.")

                else:
                    for job in saved_notes[:20]:

                        job_id = job.get("id")
                        note_type = job.get(
                            "note_type",
                            "Smart Notes"
                        )
                        status = job.get(
                            "status",
                            "unknown"
                        )

                        if status == "completed":
                            button_label = (
                                f"📝 {note_type}"
                            )
                        elif status in {"pending", "running"}:
                            button_label = (
                                f"⏳ {note_type}"
                            )
                        else:
                            button_label = (
                                f"⚠️ {note_type}"
                            )

                        if st.button(
                            button_label,
                            key=f"sidebar_history_note_{job_id}",
                            use_container_width=True
                        ):
                            st.session_state.current_note_job_id = job_id
                            st.session_state.main_navigation = (
                                "📝 Smart Notes"
                            )
                            st.rerun()

                        created = job.get(
                            "created_at",
                            ""
                        )

                        st.caption(
                            _history_date(created)
                        )

        st.markdown(
            '<div style="min-height:55px;"></div>',
            unsafe_allow_html=True
        )

        if st.button(
            "🚪 Sign Out",
            use_container_width=True,
            key="signout_button"
        ):
            clear_user_session()

    # ========================================================
    # DASHBOARD
    # ========================================================

    if choice == "🏠 Dashboard":

        st.html(f"""
        <div class="dashboard-page">

            <div class="dashboard-header">
                <div class="dark-kicker">SMART CLASSROOM AI</div>
                <h1>👋 Welcome Back, {user_name}</h1>
                <p>{role} Dashboard • Your intelligent learning workspace.</p>
            </div>

            <div class="dark-section">
                <h3>✨ Everything you need to learn smarter</h3>
                <p>
                    Upload your study material, ask questions, create notes,
                    practice quizzes and track your learning progress —
                    all from one clean workspace.
                </p>
            </div>

        </div>
        """)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.html(f"""
            <div class="dark-card">
                <div class="label">📚 MATERIALS</div>
                <div class="value">{len(st.session_state.uploaded_files)}</div>
                <div class="hint">Saved learning files</div>
            </div>
            """)

        with col2:
            st.html(f"""
            <div class="dark-card">
                <div class="label">🧩 TEXT CHUNKS</div>
                <div class="value">{len(st.session_state.rag.text_chunks)}</div>
                <div class="hint">Indexed for AI</div>
            </div>
            """)

        with col3:
            try:
                db.cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM quiz_scores
                    WHERE student=?
                    """,
                    (user_name,)
                )
                quiz_count = db.cursor.fetchone()[0]
            except Exception:
                quiz_count = 0

            st.html(f"""
            <div class="dark-card">
                <div class="label">🎯 QUIZZES</div>
                <div class="value">{quiz_count}</div>
                <div class="hint">Attempts recorded</div>
            </div>
            """)

        with col4:
            try:
                db.cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM study_plans
                    WHERE student=?
                    AND status='Pending'
                    """,
                    (user_name,)
                )
                pending_tasks = db.cursor.fetchone()[0]
            except Exception:
                pending_tasks = 0

            st.html(f"""
            <div class="dark-card">
                <div class="label">📅 PENDING TASKS</div>
                <div class="value">{pending_tasks}</div>
                <div class="hint">Study planner</div>
            </div>
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.html("""
            <div class="dark-section">
                <h3>⚡ Quick Actions</h3>
                <div class="action-item">📚 Upload learning materials</div>
                <div class="action-item">🤖 Ask questions from your documents</div>
                <div class="action-item">📝 Generate personalized notes</div>
                <div class="action-item">🎯 Practice with AI quizzes</div>
            </div>
            """)

        with col2:
            st.html("""
            <div class="dark-section">
                <h3>🧠 Clean History System</h3>
                <p>
                    Your previous chats, notes and quiz attempts are saved
                    securely in the background.
                </p>
                <p>
                    They never clutter the dashboard or active learning pages.
                    Open <b>🕘 History</b> only when you want to search or revisit them.
                </p>
            </div>
            """)

        if st.session_state.material_uploaded:
            st.html("""
            <div class="status-ready">
                ✓ Learning material is ready for AI assistance.
            </div>
            """)
        else:
            st.html("""
            <div class="status-empty">
                ○ No learning material uploaded yet. Start by opening Learning Material.
            </div>
            """)

    elif choice == "📚 Learning Material":
        upload_learning_material()

    elif choice == "📝 Smart Notes":
        notes_summarizer()

    elif choice == "🎯 Quiz Generator":
        quiz_section()

    elif choice == "📸 Attendance":
        attendance_section()

    elif choice == "📊 Analytics":
        analytics_section()

    elif choice == "📅 Study Planner":
        study_planner()


# Recover unfinished notes jobs whenever this process starts/reruns.
# Jobs are also resumed when the user logs in.

def resume_all_saved_note_jobs():
    users_root = PERSISTENT_ROOT / "users"
    if not users_root.exists():
        return
    for user_dir in users_root.iterdir():
        jobs_dir = user_dir / "notes_jobs"
        if not jobs_dir.exists():
            continue
        for path in jobs_dir.glob("*.json"):
            job = read_json(path, {})
            if job.get("status") in {"pending", "running"} and job.get("id") and job.get("user_email"):
                active = get_active_note_jobs()
                if job["id"] not in active:
                    active.add(job["id"])
                    get_note_executor().submit(
                        run_note_job, job["id"], job["user_email"]
                    )


resume_all_saved_note_jobs()

# ============================================================
# MAIN APPLICATION
# ============================================================

if st.session_state.user:

    dashboard()

else:

    login_page()