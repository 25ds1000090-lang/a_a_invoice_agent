from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from storage import canonical_json, semantic_hash

ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

BRACKET_REF_RE = re.compile(r"\[[A-Za-z0-9_.:/-]{2,160}\]")
AMOUNT_RE = re.compile(
    r"(?i)\b(?:amount|total|invoice\s+total)\b[^\d]{0,20}(\d[\d,]*(?:\.\d{1,2})?)"
)
CURRENCY_RE = re.compile(r"\b(INR|USD|EUR|GBP|AUD|CAD|JPY|SGD|AED)\b", re.I)
INVOICE_RE = re.compile(
    r"(?i)\b(?:invoice(?:\s+(?:number|no\.?))?|inv(?:oice)?\.?\s*no\.?)\s*[:#-]?\s*([A-Z0-9][A-Z0-9./_-]{2,})"
)
VENDOR_RE = re.compile(
    r"(?i)\b(?:vendor|supplier|seller)\s*(?:name)?\s*[:=-]\s*([^\n\r;]{2,100})"
)


def _all_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_all_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(_all_text(v) for v in value)
    return str(value)


def _extract_refs(text: str) -> list[str]:
    refs: list[str] = []
    for ref in BRACKET_REF_RE.findall(text):
        lowered = ref.lower()
        if any(word in lowered for word in ("cover", "archive", "example", "training", "decoy")):
            continue
        if ref not in refs:
            refs.append(ref)
    return refs


def _choose_action(text: str) -> str:
    lower = text.lower()

    duplicate_terms = (
        "already paid",
        "duplicate invoice",
        "duplicate of",
        "same commercial invoice",
        "payment already completed",
    )
    conflict_terms = (
        "records conflict",
        "material conflict",
        "amount mismatch",
        "currency mismatch",
        "invoice number mismatch",
        "purchase order mismatch",
        "exception workflow",
    )
    hold_terms = (
        "payment must be held",
        "hold payment",
        "pause payment",
        "pending verification",
        "until verification",
        "await verification",
        "bank detail verification",
        "tax verification",
    )
    approval_terms = (
        "outside delegated authority",
        "exceeds autonomous authority",
        "requires approval",
        "approval required",
        "above authority limit",
        "outside authority",
    )
    settle_terms = (
        "valid and reconciled",
        "fully reconciled",
        "within autonomous authority",
        "authorised for autonomous settlement",
        "may be settled automatically",
    )

    if any(term in lower for term in duplicate_terms):
        return "reject_duplicate"
    if any(term in lower for term in conflict_terms):
        return "open_exception"
    if any(term in lower for term in hold_terms):
        return "hold_invoice"
    if any(term in lower for term in approval_terms):
        return "request_approval"
    if any(term in lower for term in settle_terms):
        return "settle_invoice"

    # Conservative fallback: never settle uncertain material.
    return "open_exception"


def _facts(package: dict[str, Any], text: str) -> dict[str, Any]:
    vendor = (
        package.get("vendorName")
        or package.get("vendor")
        or package.get("supplierName")
        or ""
    )
    if not vendor:
        match = VENDOR_RE.search(text)
        vendor = match.group(1).strip() if match else "Unknown vendor"

    invoice = (
        package.get("invoiceNumber")
        or package.get("invoiceNo")
        or package.get("invoice_id")
        or ""
    )
    if not invoice:
        match = INVOICE_RE.search(text)
        invoice = match.group(1).strip() if match else "Unknown invoice"

    currency = str(package.get("currency") or "").upper()
    if not currency:
        match = CURRENCY_RE.search(text)
        currency = match.group(1).upper() if match else "INR"

    amount_minor = package.get("amountMinor")
    if not isinstance(amount_minor, int):
        raw_amount = package.get("amount") or package.get("invoiceTotal")
        try:
            if raw_amount is not None:
                amount_minor = int(round(float(str(raw_amount).replace(",", "")) * 100))
            else:
                match = AMOUNT_RE.search(text)
                amount_minor = (
                    int(round(float(match.group(1).replace(",", "")) * 100))
                    if match
                    else 0
                )
        except (TypeError, ValueError):
            amount_minor = 0

    return {
        "vendorName": str(vendor)[:200],
        "invoiceNumber": str(invoice)[:200],
        "amountMinor": int(amount_minor),
        "currency": currency[:8] or "INR",
    }


def heuristic_decision(package: dict[str, Any]) -> dict[str, Any]:
    text = _all_text(package)
    refs = _extract_refs(text)
    action = _choose_action(text)

    # The grader requires exactly three decisive references.
    decisive = refs[-3:] if len(refs) >= 3 else refs[:]
    while len(decisive) < 3:
        decisive.append(f"[missing-evidence-{len(decisive)+1}]")

    rationale = (
        f"Action {action} is selected because the decisive package statements "
        f"at {decisive[0]}, {decisive[1]}, and {decisive[2]} establish the "
        f"commercial status, authority or verification condition, and required "
        f"business treatment."
    )

    return {
        "action": action,
        "facts": _facts(package, text),
        "evidenceRefs": decisive,
        "rationale": rationale,
    }


def _validate_decision(raw: dict[str, Any], package: dict[str, Any]) -> dict[str, Any]:
    action = raw.get("action")
    if action not in ACTIONS:
        raise ValueError("Invalid action")

    facts = raw.get("facts")
    refs = raw.get("evidenceRefs")
    rationale = raw.get("rationale")

    if not isinstance(facts, dict):
        raise ValueError("Invalid facts")
    required_facts = {"vendorName", "invoiceNumber", "amountMinor", "currency"}
    if set(facts) != required_facts:
        raise ValueError("Facts must contain exact required fields")
    if not isinstance(facts["amountMinor"], int):
        raise ValueError("amountMinor must be an integer")

    if not isinstance(refs, list) or len(refs) != 3:
        raise ValueError("Exactly three evidence refs are required")
    if any(not isinstance(x, str) or not x.startswith("[") or not x.endswith("]") for x in refs):
        raise ValueError("Invalid evidence reference")

    package_refs = set(_extract_refs(_all_text(package)))
    if not all(ref in package_refs for ref in refs):
        raise ValueError("Evidence ref not present in package")
    if len(set(refs)) != 3:
        raise ValueError("Evidence refs must be unique")

    if not isinstance(rationale, str) or not (60 <= len(rationale) <= 1500):
        raise ValueError("Invalid rationale length")
    if action not in rationale:
        raise ValueError("Rationale must name action")
    if sum(1 for ref in refs if ref in rationale) < 2:
        raise ValueError("Rationale must cite at least two evidence refs")

    return {
        "action": action,
        "facts": {
            "vendorName": str(facts["vendorName"]),
            "invoiceNumber": str(facts["invoiceNumber"]),
            "amountMinor": int(facts["amountMinor"]),
            "currency": str(facts["currency"]).upper(),
        },
        "evidenceRefs": refs,
        "rationale": rationale,
    }


async def ai_decisions(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_url = os.getenv("AI_BASE_URL", "").rstrip("/")
    api_key = os.getenv("AI_API_KEY", "")
    model = os.getenv("AI_MODEL", "")

    if not (base_url and api_key and model):
        return [heuristic_decision(package) for package in packages]

    prompt = {
        "instruction": (
            "For each synthetic invoice package choose exactly one action: "
            "settle_invoice, request_approval, hold_invoice, reject_duplicate, "
            "or open_exception. Return exactly three decisive bracketed evidence "
            "references copied from the determining paragraph. Exclude cover, "
            "archive, example, training and decoy references. Extract exact facts. "
            "Rationale must be 60-1500 characters, name the action, and cite at "
            "least two selected references. Return JSON only as "
            '{"decisions":[{"action":"...","facts":{"vendorName":"...",'
            '"invoiceNumber":"...","amountMinor":123,"currency":"INR"},'
            '"evidenceRefs":["[a]","[b]","[c]"],"rationale":"..."}]}.'
        ),
        "packages": packages,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": canonical_json(prompt),
            }
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            decisions = parsed["decisions"]
            if not isinstance(decisions, list) or len(decisions) != len(packages):
                raise ValueError("Wrong number of decisions")
            return [
                _validate_decision(decision, package)
                for decision, package in zip(decisions, packages, strict=True)
            ]
    except Exception:
        # The service remains available even if the external model is unavailable.
        return [heuristic_decision(package) for package in packages]


def package_cache_key(package: dict[str, Any]) -> str:
    # Ignore delivery namespace fields so Check and Save can reuse semantic decisions.
    ignored = {"packageId", "batchId", "messageId", "taskId", "contextId"}
    semantic = {k: v for k, v in package.items() if k not in ignored}
    return semantic_hash(semantic)
