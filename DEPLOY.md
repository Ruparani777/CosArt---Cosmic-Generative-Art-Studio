Deployment guide (minimal)

Prerequisites
- Kubernetes cluster or VM with Docker
- PostgreSQL instance
- Redis instance
- AWS S3 bucket and credentials
- GitHub secrets configured for env vars

Steps (quick)
1. Build Docker images for API and frontend.
2. Push images to a registry (Docker Hub, ECR, GHCR).
3. Apply k8s manifests or use docker-compose for small deployments.

Env vars (example)
- DATABASE_URL=postgresql://user:pass@db-host:5432/cosart
- REDIS_URL=redis://redis-host:6379
- SECRET_KEY=super-secret
- STRIPE_SECRET_KEY=sk_live_...
- AWS_ACCESS_KEY_ID=...
- AWS_SECRET_ACCESS_KEY=...
- CLOUD_STORAGE_BUCKET=cosart-models

Notes
- Configure HTTPS (Let's Encrypt) for frontend and API in production.
- Use managed Postgres/Redis when possible.
