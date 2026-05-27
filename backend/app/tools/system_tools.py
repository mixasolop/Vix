import subprocess

from app.schemas.tools import ToolResult


SAFE_APP_COMMANDS: dict[str, str] = {
    "notepad": "notepad.exe",
    "notepad.exe": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "calc.exe": "calc.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "mspaint.exe": "mspaint.exe",
    "explorer": "explorer.exe",
    "explorer.exe": "explorer.exe",
}


async def launch_app(arguments: dict[str, object]) -> ToolResult:
    app_name = str(arguments.get("app_name", "")).strip().lower()
    command = SAFE_APP_COMMANDS.get(app_name)
    if command is None:
        return ToolResult(
            tool="launch_app",
            status="failed",
            output={},
            error=f"Unsupported app '{app_name}'. Allowed apps: calculator, explorer, notepad, paint.",
        )

    process = subprocess.Popen([command], close_fds=True)
    return ToolResult(
        tool="launch_app",
        status="success",
        output={
            "status": "success",
            "message": f"Launched {command}.",
            "pid": process.pid,
        },
    )
