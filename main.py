from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from decision_engine import ai_decisions, package_cache_key
from models import SendRequest, Task
from storage import Store, canonical_json, semantic_hash

A2A_MEDIA_TYPE = "application/a2a+json"
INPUT_MODE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_MODE = "application/vnd.ga5.invoice-action-proposals+json"
RESULT_MODE = "application/vnd.ga5.invoice-action-results+json"
RECEIPT_MODE = "application/vnd.ga5.invoice-action-receipts+json"

TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_FAILED",
}

app = FastAPI(
    title="A2A Invoice Action Agent",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

store = Store()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def public_base_url(request: Request) -> str:
    configured = os.getenv("A2A_BASE_URL", "").strip().rstrip("/")
    if configured:
        return configured
    return f"{request.url.scheme}://{request.url.netloc}/a2a"


def principal_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"code": code, "message": message},
        media_type=A2A_MEDIA_TYPE,
    )


def task_response(task: dict[str, Any], envelope: bool = False) -> JSONResponse:
    content = {"task": task} if envelope else task
    encoded = canonical_json(content).encode("utf-8")
    if len(encoded) > 512 * 1024:
        return error(500, "RESPONSE_TOO_LARGE", "Response exceeds protocol limit")
    return JSONResponse(content=content, media_type=A2A_MEDIA_TYPE)


def authenticate(authorization: str | None) -> tuple[str | None, JSONResponse | None]:
    if authorization is None or not authorization.startswith("Bearer "):
        return None, error(401, "UNAUTHENTICATED", "Bearer authentication required")
    token = authorization[7:]
    if not token or token != token.strip():
        return None, error(403, "INVALID_CREDENTIALS", "Invalid credentials")
    return token, None


def validate_headers(
    request: Request,
    authorization: str | None,
    a2a_version: str | None,
) -> tuple[str | None, JSONResponse | None]:
    token, auth_error = authenticate(authorization)
    if auth_error:
        return None, auth_error
    if a2a_version != "1.0":
        return None, error(400, "UNSUPPORTED_A2A_VERSION", "A2A-Version must be 1.0")

    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != A2A_MEDIA_TYPE:
        return None, error(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            f"Content-Type must be {A2A_MEDIA_TYPE}",
        )
    return token, None


def task_from_row(row: Any) -> dict[str, Any]:
    return json.loads(row["task_json"])


def find_single_part(message: dict[str, Any], media_type: str) -> dict[str, Any] | None:
    matching = [
        part for part in message.get("parts", [])
        if part.get("mediaType") == media_type
    ]
    return matching[0] if len(matching) == 1 else None


def owner_not_found(conn: Any, task_id: str, phash: str) -> JSONResponse:
    # Generic response; do not disclose whether another principal owns the ID.
    return error(404, "TASK_NOT_FOUND", "Task not found")


@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request) -> JSONResponse:
    base = public_base_url(request)
    card = {
        "name": "Invoice Action Agent",
        "description": (
            "Reconciles synthetic invoice packages, proposes one controlled "
            "business action per package, and executes only grader-approved actions."
        ),
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "supportedInterfaces": [
            {
                "url": base,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": [INPUT_MODE],
        "defaultOutputModes": [PROPOSAL_MODE, RECEIPT_MODE],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice action reconciliation",
                "description": (
                    "Reads invoice claim batches, preserves exact evidence, proposes "
                    "typed actions, and records accepted receipt-bound executions."
                ),
                "tags": ["invoice", "reconciliation", "approval", "evidence"],
                "inputModes": [INPUT_MODE],
                "outputModes": [PROPOSAL_MODE, RECEIPT_MODE],
            }
        ],
    }
    return JSONResponse(content=card, media_type="application/json")


@app.post("/message:send", include_in_schema=False)
@app.post("/a2a/message:send")
async def message_send(
    request: Request,
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(default=None, alias="A2A-Version"),
) -> JSONResponse:
    token, failure = validate_headers(request, authorization, a2a_version)
    if failure:
        return failure
    assert token is not None
    phash = principal_hash(token)

    try:
        raw = await request.json()
        parsed = SendRequest.model_validate(raw)
    except (json.JSONDecodeError, ValidationError):
        return error(400, "INVALID_REQUEST", "Malformed A2A request")

    message = parsed.message.model_dump(exclude_none=True)
    message_id = message["messageId"]
    message_digest = semantic_hash(message)

    # Continuations always identify their existing task.
    if message.get("taskId") or message.get("contextId"):
        return await process_continuation(
            phash=phash,
            message=message,
            message_digest=message_digest,
        )

    initial_part = find_single_part(message, INPUT_MODE)
    if initial_part is None:
        return error(400, "INVALID_INPUT_PART", "Exactly one invoice batch part is required")

    batch = initial_part.get("data", {})
    batch_id = batch.get("batchId")
    packages = batch.get("packages")
    if not isinstance(batch_id, str) or not batch_id:
        return error(400, "INVALID_BATCH", "batchId is required")
    if not isinstance(packages, list) or not packages:
        return error(400, "INVALID_BATCH", "packages must be a nonempty array")

    package_ids = [p.get("packageId") for p in packages if isinstance(p, dict)]
    if len(package_ids) != len(packages) or any(not isinstance(x, str) or not x for x in package_ids):
        return error(400, "INVALID_PACKAGE", "Every package requires packageId")
    if len(set(package_ids)) != len(package_ids):
        return error(400, "DUPLICATE_PACKAGE_ID", "packageId values must be unique")

    with store.transaction() as conn:
        idem = store.get_idempotency(conn, phash, message_id)
        if idem:
            if idem["message_hash"] != message_digest:
                return error(409, "IDEMPOTENCY_CONFLICT", "messageId was reused with different content")
            row = store.get_task_for_owner(conn, idem["task_id"], phash)
            if row is None:
                return error(500, "STORAGE_INCONSISTENCY", "Stored task is unavailable")
            return task_response(task_from_row(row), envelope=True)

        cached: list[dict[str, Any] | None] = []
        missing_packages: list[dict[str, Any]] = []
        missing_indexes: list[int] = []
        keys: list[str] = []

        for index, package in enumerate(packages):
            if not isinstance(package, dict):
                return error(400, "INVALID_PACKAGE", "Each package must be an object")
            key = package_cache_key(package)
            keys.append(key)
            decision = store.get_cached_decision(conn, key)
            cached.append(decision)
            if decision is None:
                missing_indexes.append(index)
                missing_packages.append(package)

    # Model work occurs outside the database write lock.
    new_decisions = await ai_decisions(missing_packages) if missing_packages else []

    decisions: list[dict[str, Any]] = list(cached)
    for index, decision in zip(missing_indexes, new_decisions, strict=True):
        decisions[index] = decision

    if any(decision is None for decision in decisions):
        return error(500, "DECISION_FAILURE", "Unable to produce all decisions")

    task_id = f"tsk_{secrets.token_urlsafe(18)}"
    context_id = f"ctx_{secrets.token_urlsafe(18)}"
    action_ids: set[str] = set()
    proposals: list[dict[str, Any]] = []

    for package, decision in zip(packages, decisions, strict=True):
        action_id = f"act_{secrets.token_urlsafe(18)}"
        while action_id in action_ids:
            action_id = f"act_{secrets.token_urlsafe(18)}"
        action_ids.add(action_id)
        proposals.append(
            {
                "packageId": package["packageId"],
                "actionId": action_id,
                "action": decision["action"],
                "facts": decision["facts"],
                "evidenceRefs": decision["evidenceRefs"],
                "rationale": decision["rationale"],
            }
        )

    now = now_iso()
    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": "TASK_STATE_INPUT_REQUIRED",
            "timestamp": now,
        },
        "history": [message],
        "artifacts": [
            {
                "artifactId": f"art_{secrets.token_urlsafe(15)}",
                "name": "invoice-action-proposals",
                "parts": [
                    {
                        "mediaType": PROPOSAL_MODE,
                        "data": {
                            "batchId": batch_id,
                            "proposals": proposals,
                        },
                    }
                ],
            }
        ],
        "metadata": {
            "batchId": batch_id,
            "policyRevision": batch.get("policyRevision"),
        },
    }

    # Validate our final object before persisting it.
    Task.model_validate(task)

    with store.transaction() as conn:
        # Recheck after model work to close the concurrent first-request race.
        idem = store.get_idempotency(conn, phash, message_id)
        if idem:
            if idem["message_hash"] != message_digest:
                return error(409, "IDEMPOTENCY_CONFLICT", "messageId was reused with different content")
            row = store.get_task_for_owner(conn, idem["task_id"], phash)
            if row is None:
                return error(500, "STORAGE_INCONSISTENCY", "Stored task is unavailable")
            return task_response(task_from_row(row), envelope=True)

        for key, decision in zip(keys, decisions, strict=True):
            store.put_cached_decision(conn, key, decision)

        store.insert_task(
            conn,
            task_id=task_id,
            context_id=context_id,
            principal_hash=phash,
            batch_id=batch_id,
            state="TASK_STATE_INPUT_REQUIRED",
            task_json=task,
            now=now,
        )
        store.insert_idempotency(
            conn,
            phash,
            message_id,
            message_digest,
            task_id,
        )

    return task_response(task, envelope=True)


async def process_continuation(
    *,
    phash: str,
    message: dict[str, Any],
    message_digest: str,
) -> JSONResponse:
    message_id = message["messageId"]
    task_id = message.get("taskId")
    context_id = message.get("contextId")

    if not task_id or not context_id:
        return error(400, "INVALID_CONTINUATION", "taskId and contextId are required")

    result_part = find_single_part(message, RESULT_MODE)
    if result_part is None:
        return error(400, "INVALID_RESULT_PART", "Exactly one result part is required")

    with store.transaction() as conn:
        idem = store.get_idempotency(conn, phash, message_id)
        if idem:
            if idem["message_hash"] != message_digest:
                return error(409, "IDEMPOTENCY_CONFLICT", "messageId was reused with different content")
            row = store.get_task_for_owner(conn, idem["task_id"], phash)
            if row is None:
                return error(500, "STORAGE_INCONSISTENCY", "Stored task is unavailable")
            return task_response(task_from_row(row), envelope=True)

        row = store.get_task_for_owner(conn, task_id, phash)
        if row is None:
            return owner_not_found(conn, task_id, phash)

        task = task_from_row(row)
        if task["contextId"] != context_id:
            return error(409, "CONTEXT_MISMATCH", "Continuation does not match task context")
        if row["state"] in TERMINAL_STATES:
            return error(409, "TASK_TERMINAL", "Terminal task cannot be changed")
        if row["state"] != "TASK_STATE_INPUT_REQUIRED":
            return error(409, "TASK_NOT_READY", "Task is not awaiting results")

        data = result_part.get("data", {})
        if data.get("batchId") != row["batch_id"]:
            return error(409, "BATCH_MISMATCH", "Continuation does not match task batch")
        results = data.get("results")
        if not isinstance(results, list):
            return error(400, "INVALID_RESULTS", "results must be an array")

        proposal_part = task["artifacts"][0]["parts"][0]["data"]
        proposals = proposal_part["proposals"]
        proposal_by_package = {p["packageId"]: p for p in proposals}

        if len(results) != len(proposals):
            return error(409, "RESULT_SET_MISMATCH", "One result is required per proposal")

        seen_packages: set[str] = set()
        executions: list[dict[str, Any]] = []

        for result in results:
            if not isinstance(result, dict):
                return error(400, "INVALID_RESULT", "Each result must be an object")
            package_id = result.get("packageId")
            if package_id in seen_packages:
                return error(409, "DUPLICATE_RESULT", "Duplicate package result")
            seen_packages.add(package_id)

            proposal = proposal_by_package.get(package_id)
            if proposal is None:
                return error(409, "PROPOSAL_MISMATCH", "Result does not match a proposal")
            if result.get("actionId") != proposal["actionId"]:
                return error(409, "ACTION_ID_MISMATCH", "Result actionId does not match")
            if result.get("action") != proposal["action"]:
                return error(409, "ACTION_MISMATCH", "Result action does not match")
            if result.get("outcome") not in {"ACCEPTED", "REJECTED"}:
                return error(400, "INVALID_OUTCOME", "Invalid result outcome")
            nonce = result.get("receiptNonce")
            if not isinstance(nonce, str) or not nonce:
                return error(400, "INVALID_RECEIPT_NONCE", "receiptNonce is required")

            if result["outcome"] == "ACCEPTED":
                executions.append(
                    {
                        "packageId": proposal["packageId"],
                        "actionId": proposal["actionId"],
                        "action": proposal["action"],
                        "receiptNonce": nonce,
                        "facts": proposal["facts"],
                        "evidenceRefs": proposal["evidenceRefs"],
                    }
                )

        now = now_iso()
        task["history"].append(message)
        task["artifacts"].append(
            {
                "artifactId": f"art_{secrets.token_urlsafe(15)}",
                "name": "invoice-action-receipts",
                "parts": [
                    {
                        "mediaType": RECEIPT_MODE,
                        "data": {
                            "batchId": row["batch_id"],
                            "executions": executions,
                        },
                    }
                ],
            }
        )
        task["status"] = {
            "state": "TASK_STATE_COMPLETED",
            "timestamp": now,
        }

        changed = store.update_task(
            conn,
            task_id=task_id,
            principal_hash=phash,
            expected_states=("TASK_STATE_INPUT_REQUIRED",),
            state="TASK_STATE_COMPLETED",
            task_json=task,
            now=now,
        )
        if not changed:
            return error(409, "TASK_RACE_LOST", "Task was changed by another operation")

        store.insert_idempotency(
            conn,
            phash,
            message_id,
            message_digest,
            task_id,
        )

    return task_response(task, envelope=True)


@app.get("/tasks/{task_id}", include_in_schema=False)
@app.get("/a2a/tasks/{task_id}")
async def get_task(
    task_id: str,
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(default=None, alias="A2A-Version"),
) -> JSONResponse:
    token, auth_error = authenticate(authorization)
    if auth_error:
        return auth_error
    if a2a_version != "1.0":
        return error(400, "UNSUPPORTED_A2A_VERSION", "A2A-Version must be 1.0")
    assert token is not None
    phash = principal_hash(token)

    with store.connection() as conn:
        row = store.get_task_for_owner(conn, task_id, phash)
        if row is None:
            return owner_not_found(conn, task_id, phash)
        return task_response(task_from_row(row))


@app.get("/tasks", include_in_schema=False)
@app.get("/a2a/tasks")
async def list_tasks(
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(default=None, alias="A2A-Version"),
) -> JSONResponse:
    token, auth_error = authenticate(authorization)
    if auth_error:
        return auth_error
    if a2a_version != "1.0":
        return error(400, "UNSUPPORTED_A2A_VERSION", "A2A-Version must be 1.0")
    assert token is not None
    tasks = store.list_tasks(principal_hash(token))
    return JSONResponse(
        content={"tasks": tasks},
        media_type=A2A_MEDIA_TYPE,
    )


@app.post("/tasks/{task_id}:cancel", include_in_schema=False)
@app.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(
    task_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
    a2a_version: str | None = Header(default=None, alias="A2A-Version"),
) -> JSONResponse:
    token, failure = validate_headers(request, authorization, a2a_version)
    if failure:
        return failure
    assert token is not None
    phash = principal_hash(token)

    # Accept either an empty object or no meaningful cancellation fields.
    try:
        body = await request.json()
        if not isinstance(body, dict):
            return error(400, "INVALID_REQUEST", "Cancellation body must be an object")
    except json.JSONDecodeError:
        return error(400, "INVALID_REQUEST", "Malformed cancellation request")

    with store.transaction() as conn:
        row = store.get_task_for_owner(conn, task_id, phash)
        if row is None:
            return owner_not_found(conn, task_id, phash)
        if row["state"] in TERMINAL_STATES:
            return error(409, "TASK_TERMINAL", "Terminal task cannot be canceled")

        task = task_from_row(row)
        now = now_iso()
        task["status"] = {
            "state": "TASK_STATE_CANCELED",
            "timestamp": now,
        }

        changed = store.update_task(
            conn,
            task_id=task_id,
            principal_hash=phash,
            expected_states=(
                "TASK_STATE_SUBMITTED",
                "TASK_STATE_WORKING",
                "TASK_STATE_INPUT_REQUIRED",
            ),
            state="TASK_STATE_CANCELED",
            task_json=task,
            now=now,
        )
        if not changed:
            return error(409, "TASK_RACE_LOST", "Task was changed by another operation")

    return task_response(task)
