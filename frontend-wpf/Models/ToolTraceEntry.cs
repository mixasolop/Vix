namespace DesktopAssistant.Frontend.Models;

public sealed class ToolTraceEntry
{
    public string Tool { get; init; } = string.Empty;

    public string Arguments { get; init; } = string.Empty;

    public string Status { get; init; } = string.Empty;
}
