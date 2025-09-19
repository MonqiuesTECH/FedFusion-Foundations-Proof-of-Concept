import streamlit as st
import pandas as pd
from pydantic import BaseModel, ValidationError, constr
from typing import Dict, List
from datetime import datetime
import hashlib, uuid, time, random, json
from pathlib import Path

st.set_page_config(page_title="FedFusion POC", layout="wide")

# =========================
# Canonical Contracts (from the PoC spec)
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
# In-memory state (acts like the "hub" + SN table)
# =========================

if "candidates" not in st.session_state:
    st.session_state.candidates: Dict[str, dict] = {}
if "idempo" not in st.session_state:
    st.session_state.idempo: Dict[str, dict] = {}
if "jira_cache" not in st.session_state:
    st.session_state.jira_cache: Dict[str, dict] = {}
if "events" not in st.session_state:
    st.session_state.events: List[dict] = []
if "dlq" not in st.session_state:
    st.session_state.dlq: List[dict] = []

def log_event(kind: str, **kv):
    st.session_state.events.append({"ts": datetime.utcnow().isoformat(), "kind": kind, **kv})

def upsert_candidate(payload: CandidateScored):
    """Idempotent ingest simulating the Integration Service → ServiceNow upsert."""
    idem = payload.idempotency_key
    if idem in st.session_state.idempo:
        return st.session_state.idempo[idem]

    rec = st.session_state.candidates.get(payload.candidate_id, {})
    rec.update({
        "candidate_id": payload.candidate_id,
        "full_name": payload.full_name,
        "email": payload.email,
        "resume_uri": payload.resume_uri,
        "scores": payload.scores.model_dump(),
        "total_score": payload.total_score,
        "rank": payload.rank,
        "fields": payload.fields.model_dump(),
        "job_id": payload.job_id,
        "status": rec.get("status", "new"),
        "jira_epic": rec.get("jira_epic"),
        "jira_tasks": rec.get("jira_tasks", []),
        "trace_id": payload.trace_id,
        "idempotency_key": payload.idempotency_key,
        "created_at": rec.get("created_at", datetime.utcnow().isoformat())
    })
    st.session_state.candidates[payload.candidate_id] = rec
    st.session_state.idempo[idem] = {"ok": True, "trace_id": payload.trace_id}
    log_event("ingest", candidate_id=payload.candidate_id, trace_id=payload.trace_id)
    return {"ok": True, "trace_id": payload.trace_id}

def simulate_jira_create(orq: OnboardingRequest):
    """Idempotent Epic + Tasks creation (simulated)."""
    idem = orq.idempotency_key
    if idem in st.session_state.jira_cache:
        return st.session_state.jira_cache[idem]

    # Simulate occasional failure → DLQ + retry path
    if random.random() < 0.05:
        st.session_state.dlq.append({
            "when": datetime.utcnow().isoformat(),
            "reason": "Jira500",
            "request": orq.model_dump()
        })
        raise RuntimeError("Simulated Jira 500")

    epic_key = f"ONB-{random.randint(1000,9999)}"
    standard_tasks = ["Provision Okta", "Create Email", "SN Role", "SF Role", "Issue Laptop", "Badge"]
    tasks = [f"{epic_key}-{i+1}" for i, _ in enumerate(standard_tasks)]
    result = {"epic": epic_key, "tasks": tasks}
    st.session_state.jira_cache[idem] = result
    log_event("jira_create", candidate_id=orq.candidate_id, epic=epic_key, tasks=len(tasks))
    return result

def approve_and_orchestrate(candidate_id: str, start_date: str):
    """Approve in UI → orchestrate onboarding → update candidate with Jira keys."""
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
            log_event("approved", candidate_id=candidate_id, epic=jira["epic"])
            return {"ok": True, "epic": jira["epic"], "tasks": jira["tasks"], "trace_id": orq.trace_id, "tries": tries}
        except Exception as e:
            if tries >= 3:
                log_event("jira_failed", candidate_id=candidate_id, error=str(e))
                return {"ok": False, "error": str(e), "trace_id": orq.trace_id, "tries": tries}
            time.sleep(0.6 * tries)

# =========================
# Sidebar: Quick Start & DLQ
# =========================

st.sidebar.header("Quick Start")

# Load sample JSON from /data for convenience
sample_path = Path(__file__).parent / "data" / "sample_candidate.json"
if sample_path.exists():
    sample_text = sample_path.read_text()
    st.sidebar.code(json.loads(sample_text), language="json")
else:
    # Fallback sample
    sample_text = json.dumps({
        "candidate_id": "cand-ada-001",
        "full_name": "Ada Lovelace",
        "email": "ada@example.gov",
        "resume_uri": "s3://bucket/resumes/ada.pdf",
        "scores": {"experience": 0.82, "skills": 0.91, "education": 0.75, "clearance": 0.60},
        "total_score": 0.83,
        "rank": 1,
        "fields": {"skills": ["Go", "CICS", "ServiceNow"], "years_experience": 7, "location": "VA"},
        "idempotency_key": hashlib.sha256(b"resumebytes+email+REQ-2025-042").hexdigest(),
        "job_id": "REQ-2025-042",
        "trace_id": str(uuid.uuid4())
    }, indent=2)
    st.sidebar.code(json.loads(sample_text), language="json")

if st.sidebar.button("Ingest sample candidate"):
    try:
        payload = CandidateScored.model_validate_json(sample_text)
        res = upsert_candidate(payload)
        st.sidebar.success(f"Ingested (trace: {res['trace_id']})")
    except Exception as e:
        st.sidebar.error(str(e))

st.sidebar.markdown("---")
st.sidebar.subheader("DLQ (simulated)")
if st.session_state.dlq:
    st.sidebar.dataframe(pd.DataFrame(st.session_state.dlq))
else:
    st.sidebar.write("Empty")

# =========================
# Main UI: Tabs
# =========================
tab_list, tab_record, tab_dash = st.tabs(["Candidates", "Record", "Dashboards"])

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

    st.markdown("### Ingest a CandidateScored JSON")
    raw = st.text_area("Paste CandidateScored JSON", height=220, value="")
    if st.button("Validate + Ingest"):
        try:
            payload = CandidateScored.model_validate_json(raw)
            res = upsert_candidate(payload)
            st.success(f"OK — idempotent ingest complete (trace: {res['trace_id']}).")
        except ValidationError as ve:
            st.error(f"Schema error: {ve}")
        except Exception as e:
            st.error(str(e))

with tab_record:
    st.subheader("Record View + Approve")
    cand_ids = sorted(list(st.session_state.candidates.keys()))
    if not cand_ids:
        st.info("No candidates yet. Ingest one from the sidebar or the list tab.")
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
    ev = pd.DataFrame(st.session_state.events)
    if ev.empty:
        st.info("No events yet. Ingest and approve to populate dashboards.")
    else:
        st.write("Recent events")
        st.dataframe(ev.tail(50), use_container_width=True)
        st.write("Counts by kind")
        st.bar_chart(ev["kind"].value_counts())
