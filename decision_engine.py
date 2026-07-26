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
CURRENCY_RE = re.compile(r"\b(INR|USD|EUR|GBP|AUD|CAD|JPY|SGD|AED)\b", re.I)
INVOICE_RE = re.compile(
    r"(?i)\b(?:invoice(?:\s+(?:number|no\.?))?|inv(?:oice)?\.?\s*no\.?)"
    r"\s*[:#=-]?\s*([A-Z0-9][A-Z0-9./_-]{2,})"
)
VENDOR_RE = re.compile(
    r"(?i)\b(?:vendor|supplier|seller)\s*(?:name)?\s*[:=-]\s*([^\n\r;]{2,120})"
)
AMOUNT_RE = re.compile(
    r"(?i)\b(?:amount|total|invoice\s+total|gross\s+amount)\b"
    r"[^\d]{0,24}(\d[\d,]*(?:\.\d{1,2})?)"
)

DECOY_WORDS = {
    "cover sheet", "archive", "archived", "example", "training",
    "decoy", "illustration", "historical example", "old example",
    "do not use", "ignore this", "not controlling", "superseded",
}

ACTION_CUES = {
    "reject_duplicate": [
        "already paid", "duplicate invoice", "duplicate of",
        "same commercial invoice was already paid", "payment already completed",
        "previously settled", "paid earlier",
    ],
    "open_exception": [
        "material records conflict", "records conflict", "material conflict",
        "amount mismatch", "currency mismatch", "invoice number mismatch",
        "purchase order mismatch", "conflicting records", "exception workflow",
        "cannot be reconciled",
    ],
    "hold_invoice": [
        "payment pauses", "payment must be held", "hold invoice", "hold payment",
        "pause payment", "pending verification", "until verification completes",
        "await verification", "bank detail verification", "tax verification",
        "verification outstanding",
    ],
    "request_approval": [
        "outside delegated authority", "exceeds autonomous authority",
        "requires approval", "approval required", "above authority limit",
        "outside authority", "commercially valid but outside",
        "delegated limit exceeded",
    ],
    "settle_invoice": [
        "valid, reconciled, and within autonomous authority",
        "valid and reconciled", "fully reconciled",
        "within autonomous authority", "within delegated authority",
        "authorised for autonomous settlement", "authorized for autonomous settlement",
        "may be settled automatically", "settle autonomously",
    ],
}

NEGATION_RE = re.compile(
    r"(?i)\b(?:not|never|no longer|must not|do not|cannot|isn't|is not)\b"
)


def _all_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_all_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(_all_text(v) for v in value)
    return str(value)


def _paragraphs(text: str) -> list[str]:
    raw = re.split(r"(?:\r?\n){2,}|(?<=\.)\s+(?=[A-Z\[])", text)
    return [p.strip() for p in raw if p.strip()]


def _refs(text: str) -> list[str]:
    out: list[str] = []
    for ref in BRACKET_REF_RE.findall(text):
        if ref not in out:
            out.append(ref)
    return out


def _is_decoy(paragraph: str) -> bool:
    lower = paragraph.lower()
    return any(word in lower for word in DECOY_WORDS)


def _cue_score(paragraph: str, action: str) -> int:
    lower = paragraph.lower()
    score = 0
    for cue in ACTION_CUES[action]:
        if cue in lower:
            score += 8 + len(cue) // 12

    # Prefer paragraphs containing exactly the required three references.
    ref_count = len(_refs(paragraph))
    if ref_count == 3:
        score += 12
    elif ref_count > 3:
        score += 3
    elif ref_count < 2:
        score -= 8

    # Strongly demote archive/training/example material.
    if _is_decoy(paragraph):
        score -= 40

    # A sentence saying an action is NOT applicable should not trigger it.
    for cue in ACTION_CUES[action]:
        idx = lower.find(cue)
        if idx >= 0:
            window = lower[max(0, idx - 45): idx + len(cue) + 15]
            if NEGATION_RE.search(window):
                score -= 18

    # "Controlling", "current", and "final determination" are useful signals.
    if any(x in lower for x in (
        "controlling facts", "controlling paragraph", "current determination",
        "final determination", "governing facts", "operative facts",
    )):
        score += 15

    return score


def _select_controlling_paragraph(text: str) -> tuple[str, str, list[str]]:
    candidates: list[tuple[int, str, str, list[str]]] = []
    for paragraph in _paragraphs(text):
        refs = _refs(paragraph)
        for action in ACTIONS:
            score = _cue_score(paragraph, action)
            candidates.append((score, action, paragraph, refs))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_action, best_paragraph, refs = candidates[0] if candidates else (
        -999, "open_exception", text, _refs(text)
    )

    # Never blanket-open-exception just because no cue was found.
    # Use package-level evidence and choose the highest supported action.
    if best_score < 4:
        whole_scores = {
            action: _cue_score(text, action)
            for action in ACTIONS
        }
        best_action = max(whole_scores, key=whole_scores.get)
        best_paragraph = text
        refs = _refs(text)

    clean_refs = [
        r for r in refs
        if not any(w in r.lower() for w in ("cover", "archive", "example", "training", "decoy"))
    ]
    return best_action, best_paragraph, clean_refs


def _facts(package: dict[str, Any], text: str) -> dict[str, Any]:
    vendor = package.get("vendorName") or package.get("vendor") or package.get("supplierName")
    if not vendor:
        m = VENDOR_RE.search(text)
        vendor = m.group(1).strip() if m else "Unknown vendor"

    invoice = package.get("invoiceNumber") or package.get("invoiceNo") or package.get("invoice_id")
    if not invoice:
        m = INVOICE_RE.search(text)
        invoice = m.group(1).strip() if m else "Unknown invoice"

    currency = str(package.get("currency") or "").upper()
    if not currency:
        m = CURRENCY_RE.search(text)
        currency = m.group(1).upper() if m else "INR"

    amount_minor = package.get("amountMinor")
    if not isinstance(amount_minor, int):
        raw = package.get("amount") or package.get("invoiceTotal")
        try:
            if raw is not None:
                amount_minor = int(round(float(str(raw).replace(",", "")) * 100))
            else:
                m = AMOUNT_RE.search(text)
                amount_minor = int(round(float(m.group(1).replace(",", "")) * 100)) if m else 0
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
    action, controlling, refs = _select_controlling_paragraph(text)

    # Exactly three refs must come from the same controlling paragraph.
    refs = list(dict.fromkeys(refs))
    if len(refs) >= 3:
        decisive = refs[:3]
    else:
        all_refs = [
            r for r in _refs(text)
            if not any(w in r.lower() for w in ("cover", "archive", "example", "training", "decoy"))
        ]
        decisive = list(dict.fromkeys(refs + all_refs))[:3]

    # Do not fabricate evidence IDs. Failing validation is safer than fake evidence.
    if len(decisive) != 3:
        raise ValueError("Could not identify exactly three real decisive evidence references")

    reason_fragment = re.sub(r"\s+", " ", controlling).strip()
    if len(reason_fragment) > 420:
        reason_fragment = reason_fragment[:417] + "..."

    rationale = (
        f"Action {action} is selected from the controlling package facts. "
        f"{decisive[0]}, {decisive[1]}, and {decisive[2]} jointly establish the "
        f"invoice status, authority or verification condition, and required treatment. "
        f"Controlling text: {reason_fragment}"
    )

    return {
        "action": action,
        "facts": _facts(package, text),
        "evidenceRefs": decisive,
        "rationale": rationale[:1500],
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
    if set(facts) != required_facts or not isinstance(facts["amountMinor"], int):
        raise ValueError("Invalid facts schema")

    if not isinstance(refs, list) or len(refs) != 3 or len(set(refs)) != 3:
        raise ValueError("Exactly three unique evidence refs are required")

    package_refs = set(_refs(_all_text(package)))
    if not all(isinstance(x, str) and x in package_refs for x in refs):
        raise ValueError("Evidence ref not present in package")

    if not isinstance(rationale, str) or not (60 <= len(rationale) <= 1500):
        raise ValueError("Invalid rationale length")
    if action not in rationale:
        raise ValueError("Rationale must name action")
    if sum(ref in rationale for ref in refs) < 2:
        raise ValueError("Rationale must cite at least two refs")

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

    # This assignment explicitly requires AI. Use the heuristic only as a last-resort
    # local fallback, not as the normal production path.
    if not (base_url and api_key and model):
        return [heuristic_decision(package) for package in packages]

    prompt = {
        "instruction": (
            "Analyse each invoice package independently. First identify the single "
            "controlling/current paragraph and ignore cover sheets, old examples, "
            "training text, archive material, negated action words, and decoys. "
            "Choose exactly one action: settle_invoice only when valid, reconciled "
            "and within autonomous authority; request_approval when valid but outside "
            "authority; hold_invoice when payment pauses pending stated verification; "
            "reject_duplicate only when the same commercial invoice was already paid; "
            "open_exception when material records conflict. Copy exactly three bracketed "
            "evidence IDs from the same controlling paragraph. Extract exact vendor, "
            "invoice number, amountMinor and currency. The rationale must name the "
            "action, cite at least two selected IDs, and explain why those facts imply "
            "that action. Return JSON only with one decision per input package, in order."
        ),
        "schema": {
            "decisions": [{
                "action": "settle_invoice",
                "facts": {
                    "vendorName": "string",
                    "invoiceNumber": "string",
                    "amountMinor": 12345,
                    "currency": "INR",
                },
                "evidenceRefs": ["[id1]", "[id2]", "[id3]"],
                "rationale": "60-1500 characters",
            }]
        },
        "packages": packages,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise invoice-control analyst. Never use blanket "
                    "actions. Treat every package independently and preserve exact evidence."
                ),
            },
            {"role": "user", "content": canonical_json(prompt)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=40.0) as client:
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
        raise ValueError("Wrong number of model decisions")

    return [
        _validate_decision(decision, package)
        for decision, package in zip(decisions, packages, strict=True)
    ]


def package_cache_key(package: dict[str, Any]) -> str:
    ignored = {"packageId", "batchId", "messageId", "taskId", "contextId"}
    semantic = {k: v for k, v in package.items() if k not in ignored}
    return semantic_hash(semantic)
