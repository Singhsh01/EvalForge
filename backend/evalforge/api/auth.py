from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalforge.db import get_db
from evalforge.models import ApiToken, AuditEvent, User
from evalforge.services.auth import (
    ROLES,
    Principal,
    current_principal,
    hash_password,
    issue_token,
    require,
    token_digest,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserIn(BaseModel):
    username: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,63}$")
    password: str = Field(min_length=10, max_length=256)
    role: str


@router.post("/login")
def login(body: LoginIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    user = db.scalar(select(User).where(User.username == body.username))
    if user is None or not user.active or not verify_password(body.password, user.password_hash):
        db.add(
            AuditEvent(
                actor=body.username,
                action="auth.login_failed",
                entity_type="user",
                entity_id=body.username,
            )
        )
        db.commit()
        raise HTTPException(401, "Invalid username or password")
    token = issue_token(db, user)
    db.add(
        AuditEvent(actor=user.username, action="auth.login", entity_type="user", entity_id=user.id)
    )
    db.commit()
    return {"token": token, "user": {"username": user.username, "role": user.role}}


@router.post("/logout")
def logout(
    request: Request,
    _: Annotated[Principal, Depends(current_principal)],
    db: Session = Depends(get_db),
) -> dict[str, str]:
    raw = request.headers.get("authorization", "")[7:].strip()
    t = db.scalar(select(ApiToken).where(ApiToken.token_hash == token_digest(raw)))
    if t is not None:
        t.revoked = True
        db.commit()
    return {"status": "logged_out"}


@router.get("/me")
def me(p: Annotated[Principal, Depends(current_principal)]) -> dict[str, str]:
    return {"username": p.username, "role": p.role}


@router.get("/users")
def users(
    _: Annotated[Principal, Depends(require("admin"))], db: Session = Depends(get_db)
) -> list[dict[str, Any]]:
    return [
        {
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "active": u.active,
            "created_at": u.created_at.isoformat(),
        }
        for u in db.scalars(select(User)).all()
    ]


@router.post("/users", status_code=201)
def create_user(
    body: UserIn, p: Annotated[Principal, Depends(require("admin"))], db: Session = Depends(get_db)
) -> dict[str, str]:
    if body.role not in ROLES:
        raise HTTPException(422, f"role must be one of {ROLES}")
    if db.scalar(select(User).where(User.username == body.username)):
        raise HTTPException(409, "User exists")
    u = User(username=body.username, password_hash=hash_password(body.password), role=body.role)
    db.add(u)
    db.flush()
    db.add(
        AuditEvent(
            actor=p.username,
            action="user.created",
            entity_type="user",
            entity_id=u.id,
            data={"username": u.username, "role": u.role},
        )
    )
    db.commit()
    return {"id": u.id, "username": u.username, "role": u.role}


@router.post("/tokens/revoke")
def revoke(
    token: str, p: Annotated[Principal, Depends(current_principal)], db: Session = Depends(get_db)
) -> dict[str, str]:
    t = db.scalar(select(ApiToken).where(ApiToken.token_hash == token_digest(token)))
    if t is None or (t.user_id != p.user_id and not p.has("admin")):
        raise HTTPException(404, "Token not found")
    t.revoked = True
    db.commit()
    return {"status": "revoked"}
