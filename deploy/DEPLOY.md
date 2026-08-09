# Deploying IntegrityIQ

## Local (Docker Compose) — do this first

```bash
cp .env.example .env
# fill in WATSONX_API_KEY / WATSONX_PROJECT_ID in .env (see "IBM Cloud Lite setup" below)
docker compose up --build
```

- API: http://localhost:8000/docs (FastAPI's auto-generated Swagger UI)
- Dashboard: http://localhost:8501

## IBM Cloud Lite setup (for the watsonx.ai Granite calls)

1. Create a free IBM Cloud Lite account (you already have one from the SkillsBuild internship).
2. Create a **watsonx.ai** project and a **Watson Machine Learning** service instance (both have a free Lite plan).
3. Under the project's "Manage" tab, copy the **Project ID**.
4. Create an IBM Cloud API key: Manage → Access (IAM) → API keys → Create.
5. Put both into `.env` as `WATSONX_PROJECT_ID` and `WATSONX_API_KEY`.

## Deploying the API to IBM Cloud Code Engine (free tier)

```bash
ibmcloud login --sso
ibmcloud target -r us-south
ibmcloud ce project create --name integrity-iq

ibmcloud ce application create \
  --name integrity-iq-api \
  --build-source . \
  --build-dockerfile deploy/Dockerfile.api \
  --port 8000 \
  --env-from-secret integrity-iq-secrets \
  --min-scale 0 --max-scale 2
```

Store secrets first with:
```bash
ibmcloud ce secret create --name integrity-iq-secrets \
  --from-literal WATSONX_API_KEY=... \
  --from-literal WATSONX_PROJECT_ID=... \
  --from-literal JWT_SECRET_KEY=...
```

Point `API_BASE_URL` in the dashboard's environment at the Code Engine app's
public URL (`ibmcloud ce application get --name integrity-iq-api` prints it),
then deploy the dashboard the same way with `Dockerfile.dashboard`.

## Fallback: Render / Railway

If IBM Cloud Lite's free-tier limits are too tight for a live demo (Code
Engine's free tier does have a request/CPU-second cap), the same
`Dockerfile.api` / `Dockerfile.dashboard` deploy as-is to Render or Railway's
free tiers — useful to keep a permanently-up demo link for your resume even
if the IBM Cloud instance is scaled to zero. Keep the watsonx.ai *client*
either way, since that's the mandated Granite integration — only the
*hosting* of the API/dashboard containers would differ.
