Overview

This Proof of Concept (PoC) demonstrates FedFusion Foundations, an AI-driven workflow to automate candidate evaluation, onboarding, and compliance processes for federal contracts.

The system integrates Resume Scoring → ServiceNow → Jira Orchestration into a unified pipeline, reducing manual work, ensuring idempotency, and aligning with Anika Systems’ standards for security, compliance, and scalability.

Objectives

Eliminate manual candidate data entry in ServiceNow.

Provide hiring managers with ranked, scored candidates.

Enable one-click onboarding approval that creates Jira tasks automatically.

Ensure compliance with Anika Systems’ federal security and HR policies.

Architecture

Resume Scorer → Hub (Integration Service)

Accepts CandidateScored JSON payloads.

Performs schema validation and idempotent upsert into ServiceNow.

ServiceNow Workspace

Custom Candidate table with scoring, metadata, and status.

Flow Designer triggers onboarding orchestration.

Jira Orchestration

Approved candidates generate an Epic + Tasks (Okta, Email, Badge, etc.).

Idempotency ensures no duplicate epics.

Agent-Ready Tools (MCP-style)

Exposes candidate.ingest, onboarding.create, jira.add_comment as tools for future agent integration.

Deliverables in this PoC

Streamlit dashboard for candidate review and approval.

API mock endpoints simulating ServiceNow and Jira integrations.

Dashboards for hiring KPIs:

Candidate throughput

Approval rates

Time-to-onboarding

Security & Compliance

Idempotency keys and trace IDs on all requests.

Short-lived tokens (OIDC or scoped PATs).

PII handling per Anika Systems confidentiality and HR standards.

Compliance with Anika Systems’ at-will employment and dual-employment policies.

Next Steps

Extend Streamlit PoC into production ServiceNow and Jira APIs.

Conduct failure drills and DLQ (dead-letter queue) testing.

Prepare documentation for handoff to Anika Systems’ HRIS and IT teams.
