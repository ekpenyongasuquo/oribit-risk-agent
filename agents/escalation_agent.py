"""
agents/escalation_agent.py
---------------------------
Conjunction escalation agent.

Given a single "hit" record from the conjunction scanner, this agent:
  1. Calls an LLM (Groq -> llama-3.3-70b-versatile, or any model reachable
     via GROQ_API_KEY) to draft a short plain-language ops brief.
  2. Files that brief as a GitHub Issue on ekpenyongasuquo/nexusflow using
     the same GITHUB_TOKEN / Bearer auth pattern as the nexusflow adapter.

Environment variables (at least one LLM key required):
  GROQ_API_KEY         – preferred; routes to Groq-hosted models
  GITHUB_TOKEN         – required for issue filing (ghp_… or fine-grained PAT)
  GITHUB_REPO          – optional override, default: ekpenyongasuquo/nexusflow

CLI usage (demo / standalone):
  python agents/escalation_agent.py \\
      --station  "FREGAT DEB" \\
      --debris   "COSMOS 2251 DEB" \\
      --distance 29.0 \\
      --tca      "2026-08-13T09:42Z" \\
      --tier     WATCH
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── constants ─────────────────────────────────────────────────────────────────
GITHUB_API       = "https://api.github.com"
DEFAULT_REPO     =  os.environ.get("GITHUB_REPO", "ekpenyongasuquo/oribit-risk-agent")
GROQ_BASE        = "https://api.groq.com/openai/v1"
LLM_MODEL        = "llama-3.3-70b-versatile"

TIER_LABELS = {
    "CRITICAL": ["conjunction-critical", "space-ops"],
    "WATCH":    ["conjunction-watch",    "space-ops"],
}
TIER_ACTIONS = {
    "CRITICAL": "Initiate collision avoidance maneuver review immediately. "
                "Notify flight dynamics team and station crew.",
    "WATCH":    "Continue enhanced monitoring. "
                "Re-evaluate orbital predictions at next TLE update.",
}


# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class HitRecord:
    station:  str
    debris:   str
    distance: float          # km — closest approach
    tca:      datetime       # time of closest approach (UTC)
    tier:     str            # "CRITICAL" or "WATCH"
    source:   str = field(default="conjunction_check.py")


# ── LLM brief generation ──────────────────────────────────────────────────────
def _build_prompt(hit: HitRecord) -> str:
    action = TIER_ACTIONS.get(hit.tier, "Monitor and assess.")
    return textwrap.dedent(f"""
        You are a space operations duty officer writing a concise ops brief
        for the flight dynamics team. Use plain, direct language. No bullet
        lists. Maximum 120 words.

        Input data:
          - Protected asset : {hit.station}
          - Debris object   : {hit.debris}
          - Closest approach: {hit.distance:.1f} km
          - Time of closest approach (TCA): {hit.tca.strftime('%Y-%m-%dT%H:%MZ')}
          - Alert tier      : {hit.tier}
          - Data source     : {hit.source}

        Write the brief now. End with a single "Recommended action:" line
        that says exactly: {action}
    """).strip()


def generate_brief(hit: HitRecord) -> str:
    """
    Call the LLM via Groq and return the generated ops brief text.
    Falls back to a template string if no API key is configured.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        # Graceful fallback — still useful for testing without a key
        action = TIER_ACTIONS.get(hit.tier, "Monitor and assess.")
        return (
            f"[AUTO-BRIEF — no GROQ_API_KEY configured]\n\n"
            f"Conjunction alert ({hit.tier}): {hit.debris} passed within "
            f"{hit.distance:.1f} km of {hit.station} at "
            f"{hit.tca.strftime('%Y-%m-%dT%H:%MZ')} UTC. "
            f"Data source: {hit.source}.\n\n"
            f"Recommended action: {action}"
        )

    try:
        # openai>=1.0 supports arbitrary base_url for Groq compatibility
        from openai import OpenAI  # local import — not a hard dep for non-LLM paths
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": _build_prompt(hit)}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  [warn] LLM call failed ({exc}); falling back to template brief.",
              file=sys.stderr)
        action = TIER_ACTIONS.get(hit.tier, "Monitor and assess.")
        return (
            f"[TEMPLATE BRIEF — LLM error]\n\n"
            f"Conjunction alert ({hit.tier}): {hit.debris} will pass within "
            f"{hit.distance:.1f} km of {hit.station} at "
            f"{hit.tca.strftime('%Y-%m-%dT%H:%MZ')} UTC.\n\n"
            f"Recommended action: {action}"
        )


# ── GitHub issue filing ───────────────────────────────────────────────────────
def _issue_title(hit: HitRecord) -> str:
    tier_tag = f"[{hit.tier}]"
    return (
        f"{tier_tag} Conjunction: {hit.debris} / {hit.station} — "
        f"{hit.distance:.1f} km @ {hit.tca.strftime('%Y-%m-%dT%H:%MZ')}Z"
    )


def _issue_body(hit: HitRecord, brief: str) -> str:
    return textwrap.dedent(f"""
        ## Conjunction Alert — {hit.tier}

        | Field | Value |
        |---|---|
        | Protected asset | `{hit.station}` |
        | Debris object | `{hit.debris}` |
        | Closest approach | **{hit.distance:.1f} km** |
        | TCA (UTC) | `{hit.tca.strftime('%Y-%m-%dT%H:%MZ')}` |
        | Alert tier | **{hit.tier}** |
        | Data source | {hit.source} |

        ---

        ## Ops Brief

        {brief}

        ---
        *Filed automatically by `agents/escalation_agent.py`*
    """).strip()


def file_github_issue(hit: HitRecord, brief: str, repo: str = DEFAULT_REPO) -> dict:
    """
    Create a GitHub Issue via REST API v3.
    Returns the parsed JSON response dict (contains 'html_url', 'number', etc.)
    Raises RuntimeError on auth/permission failure.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is not set. Export it before running:\n"
            "  $env:GITHUB_TOKEN = 'ghp_...'"
        )

    owner, name = repo.split("/", 1)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload: dict = {
        "title":  _issue_title(hit),
        "body":   _issue_body(hit, brief),
        "labels": TIER_LABELS.get(hit.tier, ["space-ops"]),
    }

    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            f"{GITHUB_API}/repos/{owner}/{name}/issues",
            headers=headers,
            json=payload,
        )

    if resp.status_code == 401:
        raise RuntimeError("GitHub authentication failed — check GITHUB_TOKEN.")
    if resp.status_code == 403:
        raise RuntimeError(
            "GitHub permission denied — ensure the token has 'issues: write' scope "
            f"on {repo}."
        )
    if resp.status_code == 404:
        raise RuntimeError(f"GitHub repo {repo} not found.")
    if resp.status_code == 422:
        # Labels that don't exist yet cause 422; retry without labels
        payload.pop("labels")
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                f"{GITHUB_API}/repos/{owner}/{name}/issues",
                headers=headers,
                json=payload,
            )

    resp.raise_for_status()
    return resp.json()


# ── public entry point (used by conjunction_check.py pipeline) ────────────────
def escalate(hit: HitRecord, repo: str = DEFAULT_REPO) -> dict:
    """Generate brief + file issue. Returns the GitHub issue response dict."""
    brief  = generate_brief(hit)
    result = file_github_issue(hit, brief, repo=repo)
    return {"brief": brief, "issue": result}


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an ops brief and file a GitHub Issue for a conjunction hit."
    )
    p.add_argument("--station",  required=True, help="Protected station name")
    p.add_argument("--debris",   required=True, help="Debris object name")
    p.add_argument("--distance", required=True, type=float, help="Closest approach (km)")
    p.add_argument("--tca",      required=True,
                   help="Time of closest approach ISO-8601, e.g. 2026-08-13T09:42Z")
    p.add_argument("--tier",     required=True, choices=["CRITICAL", "WATCH"])
    p.add_argument("--repo",     default=DEFAULT_REPO,
                   help=f"GitHub repo to file issue on (default: {DEFAULT_REPO})")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print brief but do NOT file a GitHub Issue")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    tca = datetime.fromisoformat(args.tca.replace("Z", "+00:00"))
    hit = HitRecord(
        station=args.station,
        debris=args.debris,
        distance=args.distance,
        tca=tca,
        tier=args.tier,
    )

    print("\n=== Generating ops brief ===")
    brief = generate_brief(hit)
    print(brief)

    if args.dry_run:
        print("\n[dry-run] GitHub Issue NOT filed.")
        return

    print(f"\n=== Filing GitHub Issue on {args.repo} ===")
    issue = file_github_issue(hit, brief, repo=args.repo)
    print(f"  Issue #{issue['number']} created: {issue['html_url']}")


if __name__ == "__main__":
    main()
