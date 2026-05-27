import logging
import subprocess
from datetime import datetime

from app.schemas.tools import ToolResult

LOGGER = logging.getLogger("app.tools.system")


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
    LOGGER.info("launch_app requested | app_name=%s", app_name)
    command = SAFE_APP_COMMANDS.get(app_name)
    if command is None:
        LOGGER.warning("launch_app rejected | app_name=%s | reason=not_whitelisted", app_name)
        return ToolResult(
            tool="launch_app",
            status="failed",
            output={},
            error=f"Unsupported app '{app_name}'. Allowed apps: calculator, explorer, notepad, paint.",
        )

    try:
        process = subprocess.Popen([command], close_fds=True)
    except FileNotFoundError:
        LOGGER.exception("launch_app failed | command=%s | reason=file_not_found", command)
        return ToolResult(tool="launch_app", status="failed", output={}, error=f"Command was not found: {command}.")
    except PermissionError:
        LOGGER.exception("launch_app failed | command=%s | reason=permission_denied", command)
        return ToolResult(tool="launch_app", status="failed", output={}, error=f"Permission denied while launching {command}.")
    except OSError as exc:
        LOGGER.exception("launch_app failed | command=%s | reason=os_error", command)
        return ToolResult(tool="launch_app", status="failed", output={}, error=f"Failed to launch {command}: {exc}")

    LOGGER.info("launch_app started process | command=%s | pid=%s", command, process.pid)
    return ToolResult(
        tool="launch_app",
        status="success",
        output={
            "status": "success",
            "message": f"Launched {command}.",
            "pid": process.pid,
        },
    )


async def get_current_time(arguments: dict[str, object]) -> ToolResult:
    now = datetime.now().astimezone()
    timezone = now.tzname() or "local time"
    readable_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return ToolResult(
        tool="get_current_time",
        status="success",
        output={
            "status": "success",
            "message": f"Current time is {readable_time} {timezone}.",
            "iso_time": now.isoformat(),
            "timezone": timezone,
        },
    )
