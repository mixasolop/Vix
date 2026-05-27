namespace DesktopAssistant.Frontend.Models;

public sealed class ChatMessage
{
    public string Speaker { get; init; } = string.Empty;

    public string Text { get; init; } = string.Empty;
}
