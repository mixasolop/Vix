from app.schemas.tools import ToolResult


def fake_launch_app(app_name: str) -> ToolResult:
    return ToolResult(
        tool="launch_app",
        status="success",
        output={
            "status": "success",
            "message": f"Fake launch completed for {app_name}.",
        },
    )
