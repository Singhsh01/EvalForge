# Security architecture

## Roles
| Role | Can |
|---|---|
| viewer | read everything; ask the Copilot; call inference |
| engineer | upload data, train, register, run gates/pipelines/evals, deploy to staging, file requests |
| approver | everything an engineer can, plus decide approvals filed by **someone else** |
| admin | manage users; approve |

## Controls
| Area | Control |
|---|---|
| Authentication | Opaque bearer tokens (32 random bytes), stored as SHA-256 only, 12 h TTL, revocable (`/api/auth/logout`); PBKDF2-SHA256 (240k) passwords |
| Bootstrap | First start creates admin/engineer/approver/viewer from `EVALFORGE_BOOTSTRAP_PASSWORD`; in `env=prod` the API refuses to start without it |
| Authorisation | Role checks per endpoint; separation of duties on approvals (`EVALFORGE_ALLOW_SELF_APPROVAL=false`) |
| Production change control | Promotion, production deploy, canary promotion, production rollback, retraining, Copilot-requested evaluations and prompt releases all require an approval; gates are re-checked at decision time |
| Rate limiting | Sliding window per credential/IP: login 10/min, writes 240/min, serving 1200/min (configurable) |
| Input validation | Pydantic models everywhere; upload size/row/column limits; format allow-list; schema-compatibility for new data versions; hyperparameter schemas; prompt templates use `{{var}}` substitution only |
| Untrusted artifacts | Uploaded files are parsed, never executed or unpickled; models are produced only by EvalForge and stored with **skops** so loading never runs pickled code |
| Secrets | Provider keys are read from env vars only, never returned by the API, and redacted from traces and logs |
| Agent safety | Tool allow-list per agent config; read/action separation; actions only file approvals; step/time budgets; cancellation; evidence validation |
| Audit | Login (success/failure), user creation, every governed action and decision, with actor and payload |
| Network | Compose binds all ports to 127.0.0.1; containers run as non-root |

## Automated security testing
`tests/test_ops.py` (401/403 paths, rate limiting, token revocation, separation of duties,
gate enforcement), `tests/test_observability.py` (redaction), CI: bandit, pip-audit, npm audit.

## Not in scope yet
SSO/OIDC, MFA, per-dataset access control, encryption at rest beyond the database's own,
Redis-backed rate limiting for multiple API replicas, telemetry retention jobs.
