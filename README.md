# Orbital Risk Triage Agent

AI Builders Challenge with IBM Bob — August 2026 — Advance Space Exploration with AI

## Problem Statement
Space debris collision risk assessment is normally the domain of specialized agencies with
access to expensive tracking infrastructure and proprietary conjunction-assessment software.
Smaller mission teams, researchers, and students have no accessible way to get a plain-language,
prioritized read on whether a debris field poses a real risk to a specific spacecraft — public
orbital data exists, but turning it into an actionable alert requires orbital mechanics
expertise most teams don't have in-house.

## Solution Description
This agent ingests live orbital element data for real historical debris fields (the 2009
Cosmos 2251 / Iridium 33 collision) and a curated list of currently active crewed stations
and vehicles, computes real closest-approach risk using deterministic orbital propagation
(not LLM guesswork), and escalates only genuinely risky conjunctions as plain-language ops
briefs filed automatically as GitHub Issues.

## AI Approach and Architecture
- **Ingest layer** (`risk_engine/conjunction_check.py`, data via `cache/*.json`): pulls OMM
  orbital element data from CelesTrak's public GP API for two independent real debris
  fields (Cosmos 2251: 594 objects, Iridium 33: 111 objects) and a curated list of 13
  protected station-class assets (ISS modules, CSS modules, and currently docked/en-route
  crew and cargo vehicles), filtered down from CelesTrak's broader `stations` group which
  also returns unrelated debris and small rideshare payloads.
- **Risk engine (parse-before-LLM)**: uses the `sgp4` library to propagate every object's
  position across a 7-day window at 1-minute resolution, vectorised with NumPy
  (`SatrecArray` batch propagation) for performance. Closest-approach distance is computed
  deterministically for every debris-vs-station pair — no LLM is involved in detection.
  This is the core differentiator: risk classification is never hallucinated, only the
  final plain-language summary is AI-generated.
- **Tiered alerting**: CRITICAL (<5 km) and WATCH (<25 km) thresholds, based on standard
  conjunction-assessment warning bands.
- **Escalation agent** (`agents/escalation_agent.py`): on a flagged conjunction, calls a
  Groq-hosted LLM (`llama-3.3-70b-versatile`) to turn the structured risk data into a
  concise, plain-language ops brief, then files it as a GitHub Issue via the REST API,
  labeled by alert tier. Falls back gracefully to a template brief if no LLM key is
  configured, so the pipeline never breaks on missing credentials.

## Selected Challenge Theme
Advance Space Exploration with AI — Space debris tracking and collision avoidance systems.

## How IBM Bob Was Used
Bob was used as the primary development tool throughout: writing and iterating the CelesTrak
ingest logic against the real API response schema, building the `sgp4`-based conjunction
engine and its NumPy vectorisation (reducing a 7-day/1-minute-resolution scan from an
impractical multi-minute runtime to under 40 seconds), writing and debugging the self-test
suite that validated the engine against a known real close approach, and building the
Groq-integrated escalation agent including its GitHub Issue filing logic.

## Proof Metric
The engine was validated against a real, known close approach before being trusted on live
data: a self-test confirmed the system correctly detects a genuine 18–29 km minimum approach
between Cosmos 2251 and Fregat debris fragments (cross-checked at two different points in the
build), proving the detection logic works and isn't silently returning false negatives.
Running the full 7-day, 1-minute-resolution scan against both real debris fields (705 objects
combined) and all 13 protected station-class assets found 0 CRITICAL and 0 WATCH conjunctions
in the live window — a legitimate "all clear" result, consistent with the fact that active
stations actively maneuver to avoid exactly this kind of close approach.

## Setup
```bash
pip install -r requirements.txt

# Fetch fresh orbital data
python -c "from risk_engine.conjunction_check import *"  # or run the fetch steps in ingest/

# Run the 7-day conjunction scan
python risk_engine/conjunction_check.py

# Validate the engine against a known close approach
python risk_engine/test_selfcheck.py

# Demo the escalation agent standalone (dry run, no GitHub Issue filed)
python agents/escalation_agent.py --dry-run \
    --station "ISS (ZARYA)" --debris "COSMOS 2251 DEB" \
    --distance 29 --tca "2026-08-08T00:27Z" --tier WATCH
```

## Environment Variables
Create a `.env` file (never committed — see `.gitignore`) with:
```
GROQ_API_KEY=your_groq_key
GITHUB_TOKEN=your_github_token
GITHUB_REPO=your_username/your_repo_name
```