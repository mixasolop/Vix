# Desktop Assistant

Stage 1 scaffold for a Windows-only desktop assistant.

## What exists now

- `frontend-wpf/`: WPF shell that starts idle, accepts user text, posts runs, and renders backend WebSocket events.
- `backend/`: FastAPI backend with accepted-run `/chat`, live event stream, tool registry/runtime, permission routes, and SQLite trace storage.

The WPF app does not plan, execute tools, or store memory. It only sends user decisions to Python and displays backend event state.

## Frontend

Build the WPF shell:

```powershell
dotnet build .\DesktopAssistant.sln
```

Run the WPF shell:

```powershell
dotnet run --project .\frontend-wpf\frontend-wpf.csproj
```

## Backend

Create a virtual environment with Python 3.12 if available:

```powershell
py -3.12 -m venv .\backend\.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install -e ".\backend[dev]"
```

Run the API:

```powershell
cd .\backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Endpoints included:

- `GET /health`
- `POST /chat`
- `GET /tools`
- `POST /permissions/{permission_id}/approve`
- `POST /permissions/{permission_id}/reject`
- `WS /ws/events`

Run backend tests:

```powershell
cd .\backend
pytest
```
