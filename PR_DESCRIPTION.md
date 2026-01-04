PR: Production readiness updates (auth, rate-limiting, payments, cloud, monitoring, migrations)

Summary
-------
This branch (`prod-updates`) contains the initial production-readiness changes:
- Full authentication stack (JWT, password hashing, tiered access)
- Rate limiting with SlowAPI + Redis support
- Stripe payment integration scaffolding
- AWS S3 cloud storage service for images and models
- Monitoring and structured logging (Prometheus + structlog)
- Alembic migrations and a starter migration
- CI workflow (GitHub Actions)

Notes
-----
- Sensitive values are not committed (placeholders used). Set secrets in GitHub:
  - `DATABASE_URL` (Postgres), `REDIS_URL`, `SECRET_KEY`
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `CLOUD_STORAGE_BUCKET`
  - `STRIPE_SECRET_KEY`, `STRIPE_PRO_PRICE_ID`, `STRIPE_STUDIO_PRICE_ID`
  - SMTP credentials for email

Testing instructions
--------------------
1. Create a Python virtualenv and install requirements.
2. For quick dev, use SQLite (no `DATABASE_URL`) and run migrations via Alembic.
3. Start backend: `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
4. Start frontend: `cd frontend && npm install && npm start`

Recommended next PRs
--------------------
- Wire real model weights and generator initialization
- Add end-to-end tests and fixtures
- Harden payment webhooks and email flows
- Add production deployment manifests (k8s / Terraform)
