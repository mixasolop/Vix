using System.Text.Json.Serialization;

namespace DesktopAssistant.Frontend.Dtos;

public sealed class ChatRequestDto
{
    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("conversation_id")]
    public string? ConversationId { get; set; }
}

public sealed class ChatAcceptedDto
{
    [JsonPropertyName("accepted")]
    public bool Accepted { get; set; }

    [JsonPropertyName("conversation_id")]
    public string ConversationId { get; set; } = string.Empty;

    [JsonPropertyName("run_id")]
    public string RunId { get; set; } = string.Empty;
}

public sealed class PlanDto
{
    [JsonPropertyName("goal")]
    public string Goal { get; set; } = string.Empty;

    [JsonPropertyName("steps")]
    public List<PlanStepDto> Steps { get; set; } = [];
}

public sealed class PlanStepDto
{
    [JsonPropertyName("number")]
    public int Number { get; set; }

    [JsonPropertyName("title")]
    public string Title { get; set; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;
}

public sealed class ToolCallDto
{
    [JsonPropertyName("tool")]
    public string Tool { get; set; } = string.Empty;

    [JsonPropertyName("arguments")]
    public Dictionary<string, object?> Arguments { get; set; } = [];

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("result")]
    public Dictionary<string, object?> Result { get; set; } = [];

    [JsonPropertyName("error")]
    public string? Error { get; set; }
}

public sealed class ToolListResponseDto
{
    [JsonPropertyName("tools")]
    public List<ToolDefinitionDto> Tools { get; set; } = [];
}

public sealed class ToolDefinitionDto
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;
}

public sealed class AssistantEventDto
{
    [JsonPropertyName("event_id")]
    public string EventId { get; set; } = string.Empty;

    [JsonPropertyName("session_id")]
    public string? SessionId { get; set; }

    [JsonPropertyName("run_id")]
    public string? RunId { get; set; }

    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;

    [JsonPropertyName("data")]
    public Dictionary<string, object?> Data { get; set; } = [];

    [JsonPropertyName("timestamp")]
    public DateTimeOffset Timestamp { get; set; }
}
