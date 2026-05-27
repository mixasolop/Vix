namespace DesktopAssistant.Frontend.Models;

public sealed class PlanStep
{
    public int Number { get; init; }

    public string Title { get; init; } = string.Empty;

    public string DisplayText => $"{Number}. {Title}";
}
