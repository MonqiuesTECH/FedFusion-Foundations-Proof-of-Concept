# app.py
import streamlit as st
import pandas as pd
from pydantic import BaseModel, ValidationError, constr
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import hashlib, uuid, time, random, re, io, json, os, sqlite3

# ---- resume parsers
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

st.set_page_config(page_title="FedFusion POC", layout="wide")

# =========================
# Config — tweak for your domain
# =========================
MAX_FILES = 10
DEFAULT_JOB_ID = "REQ-POC-0001"

# Rule-based scoring for a transparent POC
SKILL_KEYWORDS = {
    "servicenow": 3.0, "jira": 2.0, "python": 2.5, "sql": 1.5,
    "docker": 1.8, "kubernetes": 2.2, "aws": 2.0, "gcp": 1.7, "azure": 1.7,
    "microservices": 1.5, "ml": 1.5, "ai": 1.5, "rag": 1.7, "langchain": 1.2,
    "compliance": 1.0, "hipaa": 1.0, "gdpr": 0.8, "soc2": 0.8
}
EDU_KEYWORDS = {"phd":1.0,"master":0.7,"ms ":0.7,"m.s.":0.7,"bachelor":0.5,"bs ":0.5,"b.s.":0.5,"associate":0.3,"associates":0.3}
CLEARANCE_KEYWORDS = {"public trust":0.5,"secret":0.7,"top secret":0.9,"ts":0.9,"ts/sci":1.0,"sci":1.0}
WEIGHTS = {"experience":0.35,"skills":0.40,"education":0.10,"clearance":0.15}

# =========================
# Data models
# =========================
class Scores(BaseModel):
    experience: float
    skills: float
    education: float
    clearance: float

class Fields(BaseModel):
    skills: List[str]
    years_experience: int
    location: str

class CandidateScored(BaseModel):
    candidate_id: str
    full_name: str
    email: str
    resume_uri: str
    scores: Scores
    total_score: float
    rank: int
    fields: Fields
    idempotency_key: str
    job_id: str
    trace_id: str

class OnboardingRequest(BaseModel):
    candidate_id: str
    job_id: str
    offer_status: constr(strip_whitespace=True) = "accepted"
    start_date: constr(strip_whitespace=True)
    access_profile: List[str]
    equipment: List[str]
    trace_id: str
    idempotency_key: str

# =========================
# SQLite persistence
# =========================
DB_DIR = Path("data")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "fedfusion.db"

def db_conn():
    return sqlite3.connect(DB_PATH)

def db_init():
    with db_conn() as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS candidates(
                candidate_id TEXT PRIMARY KEY,
                full_name TEXT, email TEXT, resume_uri TEXT,
                scores_json TEXT, total_score REAL, rank INTEGER,
                fields_json TEXT, job_id TEXT, status TEXT,
                jira_epic TEXT, jira_tasks_json TEXT,
                trace_id TEXT, idempotency_key TEXT, created_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS idempotency(
                idempotency_key TEXT PRIMARY KEY,
                result_json TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS events(
                ts TEXT, kind TEXT, payload_json TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dlq(
                ts TEXT, reason TEXT, payload_json TEXT
            )
        """)
        con.commit()

def db_upsert_candidate(c: CandidateScored):
    with db_conn() as con:
        cur = con.cursor()
        cur.execute("""
            INSERT INTO candidates(candidate_id, full_name, email, resume_uri, scores_json, total_score, rank,
                                   fields_json, job_id, status, jira_epic, jira_tasks_json, trace_id, idempotency_key, created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                full_name=excluded.full_name,
                email=excluded.email,
                resume_uri=excluded.resume_uri,
                scores_json=excluded.scores_json,
                total_score=excluded.total_score,
                rank=excluded.rank,
                fields_json=excluded.fields_json,
                job_id=excluded.job_id,
                status=COALESCE(candidates.status,'new'),
                trace_id=excluded.trace_id,
                idempotency_key=excluded.idempotency_key
        """, (
            c.candidate_id, c.full_name, c.email, c.resume_uri,
            json.dumps(c.scores.model_dump()), c.total_score, c.rank,
            json.dumps(c.fields.model_dump()), c.job_id, "new", None, json.dumps([]),
            c.trace_id, c.idempotency_key, datetime.utcnow().isoformat()
        ))
        con.commit()

def db_set_idempo(key: str, result: dict):
    with db_conn() as con:
        con.execute("INSERT OR IGNORE INTO idempotency(idempotency_key, result_json) VALUES(?,?)",
                    (key, json.dumps(result)))
        con.commit()

def db_get_idempo(key: str) -> Optional[dict]:
    with db_conn() as con:
        cur = con.execute("SELECT result_json FROM idempotency WHERE idempotency_key=?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

def db_update_jira(candidate_id: str, epic: str, tasks: List[str], status: str):
    with db_conn() as con:
        con.execute("""
            UPDATE candidates SET jira_epic=?, jira_tasks_json=?, status=?
            WHERE candidate_id=?
        """, (epic, json.dumps(tasks), status, candidate_id))
        con.commit()

def db_load_all_candidates() -> Dict[str, dict]:
    with db_conn() as con:
        cur = con.execute("SELECT candidate_id, full_name, email, resume_uri, scores_json, total_score, rank, fields_json, job_id, status, jira_epic, jira_tasks_json, trace_id, idempotency_key, created_at FROM candidates")
        out = {}
        for row in cur.fetchall():
            out[row[0]] = {
                "candidate_id": row[0], "full_name": row[1], "email": row[2], "resume_uri": row[3],
                "scores": json.loads(row[4]), "total_score": row[5], "rank": row[6],
                "fields": json.loads(row[7]), "job_id": row[8], "status": row[9],
                "jira_epic": row[10], "jira_tasks": json.loads(row[11]),
                "trace_id": row[12], "idempotency_key": row[13], "created_at": row[14]
            }
        return out

def db_log_event(kind: str, payload: dict):
    with db_conn() as con:
        con.execute("INSERT INTO events(ts, kind, payload_json) VALUES(?,?,?)",
                    (datetime.utcnow().isoformat(), kind, json.dumps(payload)))
        con.commit()

def db_log_dlq(reason: str, payload: dict):
    with db_conn() as con:
        con.execute("INSERT INTO dlq(ts, reason, payload_json) VALUES(?,?,?)",
                    (datetime.utcnow().isoformat(), reason, json.dumps(payload)))
        con.commit()

def db_load_events_df() -> pd.DataFrame:
    with db_conn() as con:
        return pd.read_sql_query("SELECT * FROM events", con)

def db_load_dlq_df() -> pd.DataFrame:
    with db_conn() as con:
        return pd.read_sql_query("SELECT * FROM dlq", con)

# =========================
# Session state cache (mirrors DB for fast UI)
# =========================
def init_state():
    if "candidates" not in st.session_state:
        st.session_state.candidates = {}
    if "idempo" not in st.session_state:
        st.session_state.idempo = {}
    if "jira_cache" not in st.session_state:
        st.session_state.jira_cache = {}

def load_from_db():
    st.session_state.candidates = db_load_all_candidates()
    # idempotency cache is only for current run; DB also stores results to prevent duplicates across restarts
    st.session_state.idempo = {}

def log_event(kind: str, **kv):
    db_log_event(kind, kv)

# =========================
# Resume parsing & scoring
# =========================
def read_docx_bytes(b: bytes) -> str:
    f = io.BytesIO(b)
    doc = Document(f)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf_bytes(b: bytes) -> str:
    return pdf_extract_text(io.BytesIO(b)) or ""

def read_txt_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return b.decode("latin-1", errors="ignore")

def extract_text_from_upload(upload) -> Tuple[str, str]:
    name_guess = Path(upload.name).stem.replace("_"," ").replace("-"," ").strip()
    suffix = upload.name.lower().split(".")[-1]
    raw = upload.read()
    if suffix == "pdf":
        text = read_pdf_bytes(raw)
    elif suffix == "docx":
        text = read_docx_bytes(raw)
    elif suffix == "txt":
        text = read_txt_bytes(raw)
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    full_name = first_line if 2 <= len(first_line.split()) <= 5 and len(first_line) <= 100 else name_guess
    return text, full_name

def extract_years_experience(text: str) -> int:
    m = re.findall(r"(\d+)\s*(\+)?\s*(years|yrs|yr)", text.lower())
    years = [int(g[0]) for g in m if g and g[0].isdigit()]
    return max(years) if years else 0

def detect_location(text: str) -> str:
    states = r"\b(AL|AK|AZ|AR|CA|CO|CT|DC|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MS|MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV|WY)\b"
    m = re.search(states, text.upper())
    return m.group(0) if m else "Unknown"

def collect_skills(text: str) -> List[str]:
    found = set()
    tl = text.lower()
    for k in SKILL_KEYWORDS.keys():
        if k in tl:
            found.add(k)
    return sorted(found)

def score_skills(text: str) -> float:
    tl = text.lower()
    score = 0.0
    for k, w in SKILL_KEYWORDS.items():
        if k in tl:
            score += w
    return min(score / 10.0, 1.0)  # normalize 0..1

def score_education(text: str) -> float:
    tl = text.lower()
    score = 0.0
    for k, w in EDU_KEYWORDS.items():
        if k in tl:
            score = max(score, w)
    return min(score, 1.0)

def score_clearance(text: str) -> float:
    tl = text.lower()
    score = 0.0
    for k, w in CLEARANCE_KEYWORDS.items():
        if k in tl:
            score = max(score, w)
    return min(score, 1.0)

def score_experience(years: int) -> float:
    return min(years / 15.0, 1.0)  # saturate at 15 years

def total_score(parts: Scores) -> float:
    return round(
        WEIGHTS["experience"] * parts.experience +
        WEIGHTS["skills"] * parts.skills +
        WEIGHTS["education"] * parts.education +
        WEIGHTS["clearance"] * parts.clearance, 4
    )

def make_candidate_from_resume(upload, job_id: str = DEFAULT_JOB_ID) -> CandidateScored:
    text, full_name = extract_text_from_upload(upload)
    years = extract_years_experience(text)
    skills_list = collect_skills(text)
    exp = score_experience(years)
    skl = score_skills(text)
    edu = score_education(text)
    clr = score_clearance(text)

    scores = Scores(experience=exp, skills=skl, education=edu, clearance=clr)
    total = total_score(scores)

    email_guess = f"{full_name.replace(' ','.').lower()}@example.gov" if full_name else "unknown@example.gov"
    candidate_id = Path(upload.name).stem.lower()
    resume_uri = f"uploaded://{upload.name}"

    idem = hashlib.sha256(f"{candidate_id}|{job_id}|{len(text)}".encode()).hexdigest()

    cand = CandidateScored(
        candidate_id=candidate_id,
        full_name=full_name or candidate_id,
        email=email_guess,
        resume_uri=resume_uri,
        scores=scores,
        total_score=round(total, 2),
        rank=0,  # set after sorting
        fields=Fields(skills=skills_list, years_experience=years, location=detect_location(text)),
        idempotency_key=idem,
        job_id=job_id,
        trace_id=str(uuid.uuid4())
    )
    return cand

# =========================
# Core ops (idempotent ingest + approve/orchestrate)
# =========================
def upsert_candidate(payload: CandidateScored):
    # cross-run idempotency: check DB first
    prior = db_get_idempo(payload.idempotency_key)
    if prior:
        return prior
    # in-run cache (not required but fast)
    if payload.idempotency_key in st.session_state.idempo:
        return st.session_state.idempo[payload.idempotency_key]

    db_upsert_candidate(payload)
    st.session_state.candidates[payload.candidate_id] = {
        "candidate_id": payload.candidate_id,
        "full_name": payload.full_name,
        "email": payload.email,
        "resume_uri": payload.resume_uri,
        "scores": payload.scores.model_dump(),
        "total_score": payload.total_score,
        "rank": payload.rank,
        "fields": payload.fields.model_dump(),
        "job_id": payload.job_id,
        "status": "new",
        "jira_epic": None,
        "jira_tasks": [],
        "trace_id": payload.trace_id,
        "idempotency_key": payload.idempotency_key,
        "created_at": datetime.utcnow().isoformat()
    }
    result = {"ok": True, "trace_id": payload.trace_id}
    st.session_state.idempo[payload.idempotency_key] = result
    db_set_idempo(payload.idempotency_key, result)
    log_event("ingest", candidate_id=payload.candidate_id, trace_id=payload.trace_id)
    return result

def simulate_jira_create(orq: OnboardingRequest):
    cache_key = orq.idempotency_key
    if cache_key in st.session_state.jira_cache:
        return st.session_state.jira_cache[cache_key]
    # simulate occasional failure -> DLQ after retries
    if random.random() < 0.05:
        db_log_dlq("Jira500", orq.model_dump())
        raise RuntimeError("Simulated Jira 500")

    epic_key = f"ONB-{random.randint(1000,9999)}"
    standard_tasks = ["Provision Okta", "Create Email", "SN Role", "SF Role", "Issue Laptop", "Badge"]
    tasks = [f"{epic_key}-{i+1}" for i, _ in enumerate(standard_tasks)]
    result = {"epic": epic_key, "tasks": tasks}
    st.session_state.jira_cache[cache_key] = result
    log_event("jira_create", candidate_id=orq.candidate_id, epic=epic_key, tasks=len(tasks))
    return result

def approve_and_orchestrate(candidate_id: str, start_date: str):
    rec = st.session_state.candidates[candidate_id]
    rec["status"] = "approved"

    orq = OnboardingRequest(
        candidate_id=candidate_id,
        job_id=rec["job_id"],
        offer_status="accepted",
        start_date=start_date,
        access_profile=["Okta", "Email", "ServiceNow", "Salesforce"],
        equipment=["Laptop", "Badge"],
        trace_id=str(uuid.uuid4()),
        idempotency_key=hashlib.sha256(f"{candidate_id}{rec['job_id']}{start_date}".encode()).hexdigest()
    )

    tries = 0
    while True:
        tries += 1
        try:
            jira = simulate_jira_create(orq)
            rec["jira_epic"] = jira["epic"]
            rec["jira_tasks"] = jira["tasks"]
            rec["status"] = "onboarding_in_progress"
            st.session_state.candidates[candidate_id] = rec
            db_update_jira(candidate_id, jira["epic"], jira["tasks"], rec["status"])
            log_event("approved", candidate_id=candidate_id, epic=jira["epic"])
            return {"ok": True, "epic": jira["epic"], "tasks": jira["tasks"], "trace_id": orq.trace_id, "tries": tries}
        except Exception as e:
            if tries >= 3:
                db_log_dlq("JiraRetriesExhausted", orq.model_dump())
                log_event("jira_failed", candidate_id=candidate_id, error=str(e))
                return {"ok": False, "error": str(e), "trace_id": orq.trace_id, "tries": tries}
            time.sleep(0.6 * tries)

# =========================
# Boot
# =========================
db_init()
init_state()
load_from_db()

# =========================
# Sidebar — DLQ & notes
# =========================
st.sidebar.header("Queues")
dlq_df = db_load_dlq_df()
if not dlq_df.empty:
    st.sidebar.dataframe(dlq_df.tail(30), use_container_width=True, height=240)
else:
    st.sidebar.write("DLQ is empty")
st.sidebar.markdown("---")
st.sidebar.caption("POC: files parsed in-memory; data persisted to ./data/fedfusion.db")

# =========================
# Main UI
# =========================
tab_upload, tab_list, tab_record, tab_dash = st.tabs(["Upload & Score", "Candidates", "Record", "Dashboards"])

with tab_upload:
    st.subheader("Upload resumes (PDF, DOCX, or TXT)")
    uploads = st.file_uploader("Select up to 10 files", type=["pdf","docx","txt"], accept_multiple_files=True)
    job_id = st.text_input("Job ID", value=DEFAULT_JOB_ID)
    col1, col2 = st.columns([1,1])
    with col1:
        min_years = st.slider("Minimum years of experience (filter after scoring)", 0, 20, 0, 1)
    with col2:
        threshold = st.slider("Minimum total score (filter after scoring)", 0.0, 1.0, 0.0, 0.01)

    if uploads:
        if len(uploads) > MAX_FILES:
            st.error(f"Please upload {MAX_FILES} files or fewer.")
        else:
            created: List[CandidateScored] = []
            errs = []
            for u in uploads:
                try:
                    cand = make_candidate_from_resume(u, job_id=job_id)
                    created.append(cand)
                except Exception as e:
                    errs.append(f"{u.name}: {e}")

            if errs:
                st.error("\n".join(errs))

            if created:
                # Rank by total_score desc, then years desc
                created.sort(key=lambda c: (-c.total_score, -c.fields.years_experience))
                for i, c in enumerate(created, start=1):
                    c.rank = i
                    upsert_candidate(c)

                # Display scored table with filters
                rows = []
                for c in created:
                    if c.fields.years_experience < min_years: 
                        continue
                    if c.total_score < threshold:
                        continue
                    rows.append({
                        "candidate_id": c.candidate_id,
                        "name": c.full_name,
                        "job_id": c.job_id,
                        "total_score": c.total_score,
                        "rank": c.rank,
                        "years_exp": c.fields.years_experience,
                        "skills_found": ", ".join(c.fields.skills)[:120]
                    })

                if rows:
                    st.success(f"Scored {len(created)} resume(s). Showing {len(rows)} after filters.")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.warning("No candidates met the current filters.")

with tab_list:
    st.subheader("ServiceNow-like List View")
    colf1, colf2 = st.columns(2)
    with colf1:
        filter_job = st.text_input("Filter by job_id", value="")
    with colf2:
        min_score = st.slider("Min total_score", 0.0, 1.0, 0.0, 0.01)

    rows = []
    for c in st.session_state.candidates.values():
        if filter_job and c["job_id"] != filter_job:
            continue
        if c["total_score"] < min_score:
            continue
        rows.append({
            "candidate_id": c["candidate_id"],
            "name": c["full_name"],
            "job_id": c["job_id"],
            "total_score": c["total_score"],
            "rank": c["rank"],
            "status": c["status"],
            "jira_epic": c.get("jira_epic")
        })
    if rows:
        df = pd.DataFrame(sorted(rows, key=lambda r: (-r["total_score"], r["rank"])))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No candidates match the filters.")

with tab_record:
    st.subheader("Record View + Approve")
    cand_ids = sorted(list(st.session_state.candidates.keys()))
    if not cand_ids:
        st.info("No candidates yet. Upload resumes on the first tab.")
    else:
        chosen = st.selectbox("Choose candidate_id", cand_ids)
        rec = st.session_state.candidates[chosen]
        colA, colB = st.columns(2)
        with colA:
            st.write("Summary")
            st.json({k: rec[k] for k in ["candidate_id","full_name","email","job_id","status","total_score","rank"]})
            st.write("Scores")
            st.json(rec["scores"])
            st.write("Fields")
            st.json(rec["fields"])
        with colB:
            st.write("Links / Jira")
            st.json({"resume_uri": rec["resume_uri"], "jira_epic": rec.get("jira_epic"), "jira_tasks": rec.get("jira_tasks")})
            st.markdown("Approve for onboarding")
            start_date = st.date_input("Start date").isoformat()
            if st.button("Approve → Kick off Jira"):
                res = approve_and_orchestrate(chosen, start_date)
                if res["ok"]:
                    st.success(f"Created Epic {res['epic']} with {len(res['tasks'])} tasks (trace: {res['trace_id']}, tries={res['tries']}).")
                else:
                    st.error(f"Failed: {res['error']} (trace: {res['trace_id']}, tries={res['tries']}). See DLQ in sidebar.")

with tab_dash:
    st.subheader("Dashboards")
    ev = db_load_events_df()
    if ev.empty:
        st.info("No events yet. Upload and approve to populate dashboards.")
    else:
        st.write("Recent events")
        st.dataframe(ev.tail(50), use_container_width=True)
        st.write("Counts by kind")
        st.bar_chart(ev["kind"].value_counts())
