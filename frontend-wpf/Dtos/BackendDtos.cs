using System.Text.Json.Serialization;
using System.Text.Json;

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

public sealed class ProposedToolListResponseDto
{
    [JsonPropertyName("tools")]
    public List<ProposedToolDto> Tools { get; set; } = [];
}

public sealed class ProposedToolDto
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;

    [JsonPropertyName("reason")]
    public string Reason { get; set; } = string.Empty;

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("risk_level")]
    public string RiskLevel { get; set; } = string.Empty;

    [JsonPropertyName("input_schema")]
    public Dictionary<string, object?> InputSchema { get; set; } = [];

    [JsonPropertyName("output_schema")]
    public Dictionary<string, object?> OutputSchema { get; set; } = [];

    [JsonPropertyName("created_from_message")]
    public string CreatedFromMessage { get; set; } = string.Empty;

    [JsonPropertyName("created_at")]
    public string CreatedAt { get; set; } = string.Empty;

    [JsonPropertyName("updated_at")]
    public string UpdatedAt { get; set; } = string.Empty;

    public string StatusLine => $"Status: {Status}";

    public string RiskLine => $"Risk: {RiskLevel}";

    public string ReasonLine => $"Reason: {Reason}";

    public string InputSchemaText => $"Input schema: {JsonSerializer.Serialize(InputSchema)}";

    public string OutputSchemaText => $"Output schema: {JsonSerializer.Serialize(OutputSchema)}";
}

public sealed class AiStatusDto
{
    [JsonPropertyName("provider")]
    public string Provider { get; set; } = string.Empty;

    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("config_file_path")]
    public string ConfigFilePath { get; set; } = string.Empty;

    [JsonPropertyName("general_answers_enabled")]
    public bool GeneralAnswersEnabled { get; set; }

    [JsonPropertyName("proposals_enabled")]
    public bool ProposalsEnabled { get; set; }

    [JsonPropertyName("api_key_configured")]
    public bool ApiKeyConfigured { get; set; }

    [JsonPropertyName("connected")]
    public bool Connected { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("detail")]
    public string Detail { get; set; } = string.Empty;

    [JsonPropertyName("tool_execution_mode")]
    public string ToolExecutionMode { get; set; } = string.Empty;
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
