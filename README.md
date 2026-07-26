# A2A 1.0 Invoice Action Agent

FastAPI implementation of the synthetic invoice-agent assignment.

## Files

- `main.py` — HTTP routes, authentication, protocol checks and lifecycle
- `models.py` — strict A2A request/task models
- `storage.py` — SQLite persistence, idempotency and semantic decision cache
- `decision_engine.py` — optional hosted AI call plus deterministic fallback
- `requirements.txt`
- `render.yaml`

## Render deployment

1. Create a new GitHub repository.
2. Upload all files from this folder to the repository root.
3. In Render, create a **Web Service** from the repository.
4. Build command:

   ```text
   pip install -r requirements.txt
   ```

5. Start command:

   ```text
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

6. Set:

   ```text
   A2A_BASE_URL=https://YOUR-SERVICE.onrender.com/a2a
   DATABASE_PATH=/tmp/a2a_invoice_agent.sqlite3
   ```

7. Optional OpenAI-compatible provider:

   ```text
   AI_BASE_URL=https://api.openai.com/v1
   AI_API_KEY=your-api-key
   AI_MODEL=gpt-4.1-nano
   ```

   AI Pipe or another compatible provider may use a different `AI_BASE_URL`
   and model name. The URL must expose `/chat/completions`.

8. Submit this as the A2A interface base URL:

   ```text
   https://YOUR-SERVICE.onrender.com/a2a
   ```

## Important deployment note

Render's `/tmp` storage is ephemeral. It is sufficient for many single-run
graders, but a persistent disk or external PostgreSQL database is preferable
when the service must survive restarts. The assignment requires durable state,
so attach persistent storage when available and set `DATABASE_PATH` to it.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

## Required headers

All protected routes require:

```text
Authorization: Bearer any-nonempty-principal-token
A2A-Version: 1.0
```

POST routes additionally require:

```text
Content-Type: application/a2a+json
```

## Accuracy note

The protocol and storage layers are implemented independently of the model.
For best hidden-package accuracy, configure an OpenAI-compatible model. The
fallback is conservative and may not identify every hidden document pattern.


## Base URL compatibility

The application exposes both route sets:

```text
/message:send
/tasks/{id}
/tasks
/tasks/{id}:cancel
```

and:

```text
/a2a/message:send
/a2a/tasks/{id}
/a2a/tasks
/a2a/tasks/{id}:cancel
```

Recommended submission:

```text
https://YOUR-SERVICE.onrender.com
```

Set:

```text
A2A_BASE_URL=https://YOUR-SERVICE.onrender.com
```

The Agent Card will then advertise the same exact submitted base URL.
