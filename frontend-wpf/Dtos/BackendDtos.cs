using System.Text.Json.Serialization;

namespace DesktopAssistant.Frontend.Dtos;

public sealed class ChatRequestDto
{
    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("conversation_id")]
    public string? ConversationId { get; set; }
}

public sealed class ChatResponseDto
{
    [JsonPropertyName("conversation_id")]
    public string ConversationId { get; set; } = string.Empty;

    [JsonPropertyName("assistant_message")]
    public string AssistantMessage { get; set; } = string.Empty;

    [JsonPropertyName("plan")]
    public PlanDto Plan { get; set; } = new();

    [JsonPropertyName("tool_calls")]
    public List<ToolCallDto> ToolCalls { get; set; } = [];

    [JsonPropertyName("permissions")]
    public List<PermissionRequestDto> Permissions { get; set; } = [];
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
}

public sealed class PermissionRequestDto
{
    [JsonPropertyName("permission_id")]
    public string PermissionId { get; set; } = string.Empty;

    [JsonPropertyName("tool")]
    public string Tool { get; set; } = string.Empty;

    [JsonPropertyName("reason")]
    public string Reason { get; set; } = string.Empty;
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
}

public sealed class AssistantEventDto
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;
}
