Release notes — Production readiness v0.1

Highlights
- Implemented authentication and tiered access
- Added rate limiting and monitoring
- Payment and cloud storage scaffolding
- Database migrations (Alembic) and CI workflow

Breaking changes / migration
- Run `python -m alembic upgrade head` after setting `DATABASE_URL` for Postgres

How to run locally (dev)
1. Backend:
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn api.main:app --reload
```
2. Frontend:
```bash
cd frontend
npm install
npm start
```
