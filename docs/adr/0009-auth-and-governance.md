# ADR 0009 — Authentication, roles and approval-gated changes

Status: accepted (2026-11-01)

## Decision
* Opaque bearer tokens (32 random bytes), stored only as SHA-256, 12-hour expiry, revocable.
  Passwords: PBKDF2-HMAC-SHA256, 240k iterations. Stdlib only — no JWT library, no secret
  key to rotate, and revocation is a single row update.
* Roles: viewer < engineer < approver < admin, enforced per endpoint.
* **Approvals** are rows with an executor per kind (`model_promotion`, `deployment`,
  `rollback`, `canary_promotion`, `retraining`, `evaluation`, `prompt_release`). The executor
  runs only when an approver who is **not the requester** accepts; its result is stored on
  the approval, which doubles as the audit record. Executors that start background work
  return a job id that is submitted only after commit.
* Gates are re-evaluated at decision time (the champion may have changed since the request).
* In-process sliding-window rate limiting per credential/IP (login, writes, serving).

## Consequences
* Production cannot change without two people (configurable only via
  `EVALFORGE_ALLOW_SELF_APPROVAL` for solo demos).
* Multi-replica deployments need Redis-backed rate limiting and an external IdP (OIDC);
  both are additive.
