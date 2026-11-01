"""Human approval checkpoints.

Every production-changing or cost-incurring action is expressed as an ``Approval``
that an *approver* (not the requester) must accept. Accepting runs the registered
executor for that kind inside the same transaction scope and records its result,
so the approval row is also the audit record of what actually happened.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import update
from sqlalchemy.orm import Session

from evalforge.config import get_settings
from evalforge.models import Approval, AuditEvent
from evalforge.services.auth import Principal

KINDS = {
    "model_promotion",
    "deployment",
    "rollback",
    "canary_promotion",
    "retraining",
    "evaluation",
    "prompt_release",
}

Executor = Callable[[Session, Approval, Principal], dict[str, Any]]
_EXECUTORS: dict[str, Executor] = {}


class GovernanceError(ValueError):
    """Business-rule violation (mapped to HTTP 409/422)."""


def executor(kind: str) -> Callable[[Executor], Executor]:
    def deco(fn: Executor) -> Executor:
        _EXECUTORS[kind] = fn
        return fn

    return deco


def request_approval(
    db: Session,
    *,
    kind: str,
    target_type: str,
    target_id: str,
    payload: dict[str, Any],
    summary: str,
    actor: Principal,
) -> Approval:
    if kind not in KINDS:
        raise GovernanceError(f"Unknown approval kind {kind}")
    dup = db.query(Approval).filter_by(kind=kind, target_id=target_id, status="pending").first()
    if dup is not None:
        raise GovernanceError(f"An approval for this {target_type} is already pending ({dup.id})")
    a = Approval(
        kind=kind,
        target_type=target_type,
        target_id=target_id,
        payload=payload,
        summary=summary,
        requested_by=actor.username,
    )
    db.add(a)
    db.flush()
    db.add(
        AuditEvent(
            actor=actor.username,
            action="approval.requested",
            entity_type="approval",
            entity_id=a.id,
            data={"kind": kind, "target": target_id, "summary": summary},
        )
    )
    return a


def decide(
    db: Session, approval: Approval, *, approve: bool, actor: Principal, reason: str
) -> Approval:
    if approval.status != "pending":
        raise GovernanceError(f"Approval is already {approval.status}")
    if not actor.has("approver"):
        raise PermissionError("Only approvers can decide approvals")
    if approval.requested_by == actor.username and not get_settings().allow_self_approval:
        raise PermissionError("Separation of duties: you cannot approve your own request")
    # Atomically claim the approval: of two concurrent deciders only one sees rowcount 1, so
    # the executor can never run twice (audit SEC-04). On PostgreSQL the second UPDATE waits
    # for the first transaction and then re-checks the WHERE clause.
    claimed = db.execute(
        update(Approval)
        .where(Approval.id == approval.id, Approval.status == "pending")
        .values(status="deciding")
        .execution_options(synchronize_session=False)
    ).rowcount
    if claimed != 1:
        db.expire(approval, ["status"])
        raise GovernanceError(f"Approval is already {approval.status}")
    approval.decided_by, approval.decision_reason = actor.username, reason
    approval.decided_at = datetime.now(UTC)
    if not approve:
        approval.status = "rejected"
        db.add(
            AuditEvent(
                actor=actor.username,
                action="approval.rejected",
                entity_type="approval",
                entity_id=approval.id,
                data={"reason": reason},
            )
        )
        return approval
    approval.status = "approved"
    fn = _EXECUTORS.get(approval.kind)
    try:
        if fn is None:
            raise GovernanceError(f"No executor registered for {approval.kind}")
        with db.begin_nested():
            approval.result = fn(db, approval, actor)
        approval.status = "executed"
    except Exception as exc:
        approval.status = "failed"
        approval.result = {"error": f"{type(exc).__name__}: {exc}"}
    db.add(
        AuditEvent(
            actor=actor.username,
            action=f"approval.{approval.status}",
            entity_type="approval",
            entity_id=approval.id,
            data={"kind": approval.kind, "result": approval.result},
        )
    )
    return approval
