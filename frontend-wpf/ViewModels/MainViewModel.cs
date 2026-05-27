using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Windows;
using DesktopAssistant.Frontend.Commands;
using DesktopAssistant.Frontend.Dtos;
using DesktopAssistant.Frontend.Models;
using DesktopAssistant.Frontend.Services;

namespace DesktopAssistant.Frontend.ViewModels;

public sealed class MainViewModel : INotifyPropertyChanged, IDisposable
{
    private static readonly Uri BackendUri = new("http://127.0.0.1:8000");

    private readonly BackendHttpClient _backendHttpClient = new(BackendUri);
    private readonly BackendWebSocketClient _backendWebSocketClient = new(BackendUri);
    private readonly BackendProcessManager _backendProcessManager = new();
    private readonly CancellationTokenSource _shutdown = new();
    private readonly TaskCompletionSource<bool> _eventStreamReady = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private string? _conversationId;
    private string? _currentPermissionId;
    private string _backendStatus = "Disconnected";
    private string _backendStatusDetail = "Backend has not been checked yet.";
    private string _draftInput = string.Empty;
    private string _goalTitle = "No active plan";
    private string _toolTracePlaceholder = "No tool calls yet";
    private bool _isBusy;
    private bool _isReady;
    private bool _hasReceivedEvent;

    public MainViewModel()
    {
        SendCommand = new AsyncRelayCommand(SendAsync, CanSend);
        ApprovePermissionCommand = new AsyncRelayCommand(ApprovePermissionAsync, HasPendingPermission);
        RejectPermissionCommand = new AsyncRelayCommand(RejectPermissionAsync, HasPendingPermission);
        ApproveProposedToolCommand = new AsyncRelayCommand(ApproveProposedToolAsync, HasProposedToolId);
        RejectProposedToolCommand = new AsyncRelayCommand(RejectProposedToolAsync, HasProposedToolId);
        NeedsChangesProposedToolCommand = new AsyncRelayCommand(NeedsChangesProposedToolAsync, HasProposedToolId);

        PermissionItems.Add("No pending permissions");
        ImplementedTools.Add("Backend not connected");
        PlannedTools.Add("Backend not connected");
        DisabledTools.Add("Backend not connected");
        EventLogEntries.Add("No events received");
        FailedActions.Add("No failed actions");
        RecentToolCalls.Add("No tool calls yet");
        FailedToolCalls.Add("No failed tool calls");
        PermissionHistory.Add("No permission history yet");

        RefreshSettings();
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public AsyncRelayCommand SendCommand { get; }

    public AsyncRelayCommand ApprovePermissionCommand { get; }

    public AsyncRelayCommand RejectPermissionCommand { get; }

    public AsyncRelayCommand ApproveProposedToolCommand { get; }

    public AsyncRelayCommand RejectProposedToolCommand { get; }

    public AsyncRelayCommand NeedsChangesProposedToolCommand { get; }

    public string DraftInput
    {
        get => _draftInput;
        set
        {
            if (SetProperty(ref _draftInput, value))
            {
                SendCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public string GoalTitle
    {
        get => _goalTitle;
        private set => SetProperty(ref _goalTitle, value);
    }

    public string ToolTracePlaceholder
    {
        get => _toolTracePlaceholder;
        private set => SetProperty(ref _toolTracePlaceholder, value);
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (SetProperty(ref _isBusy, value))
            {
                SendCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public ObservableCollection<ChatMessage> ChatMessages { get; } = [];

    public ObservableCollection<PlanStep> PlanSteps { get; } = [];

    public ObservableCollection<ToolTraceEntry> ToolTraces { get; } = [];

    public ObservableCollection<string> PermissionItems { get; } = [];

    public ObservableCollection<string> ImplementedTools { get; } = [];

    public ObservableCollection<string> PlannedTools { get; } = [];

    public ObservableCollection<string> DisabledTools { get; } = [];

    public ObservableCollection<string> EventLogEntries { get; } = [];

    public ObservableCollection<string> FailedActions { get; } = [];

    public ObservableCollection<string> RecentToolCalls { get; } = [];

    public ObservableCollection<string> FailedToolCalls { get; } = [];

    public ObservableCollection<string> PermissionHistory { get; } = [];

    public ObservableCollection<ProposedToolDto> ProposedTools { get; } = [];

    public ObservableCollection<StatusItem> Settings { get; } = [];

    public async Task InitializeAsync()
    {
        SetBackendStatus("Starting", "Checking local backend health.");

        try
        {
            if (!await _backendHttpClient.IsHealthyAsync(_shutdown.Token))
            {
                var backendDirectory = _backendProcessManager.LocateBackendDirectory();
                _backendProcessManager.StartBackend(backendDirectory);
                await WaitForBackendAsync(_shutdown.Token);
            }

            _ = Task.Run(() => ListenForBackendEventsAsync(_shutdown.Token));
            SetBackendStatus("Connecting", "Opening WebSocket event stream.");
            await _eventStreamReady.Task.WaitAsync(TimeSpan.FromSeconds(5), _shutdown.Token);

            SetBackendStatus("Connected", "Backend is available at http://127.0.0.1:8000.");
            await LoadRegisteredToolsAsync(_shutdown.Token);
            await LoadProposedToolsAsync(_shutdown.Token);
            _isReady = true;
            SendCommand.RaiseCanExecuteChanged();
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _isReady = false;
            SendCommand.RaiseCanExecuteChanged();
            SetBackendStatus("Error", ex.Message);
            ReplaceWithSingle(FailedActions, $"Backend startup failed: {ex.Message}");
        }
    }

    public void Dispose()
    {
        _shutdown.Cancel();
        _shutdown.Dispose();
        _backendHttpClient.Dispose();
        _backendProcessManager.Dispose();
    }

    private async Task SendAsync()
    {
        var message = DraftInput.Trim();
        if (message.Length == 0)
        {
            return;
        }

        DraftInput = string.Empty;
        IsBusy = true;

        try
        {
            if (!_isReady)
            {
                ChatMessages.Add(new ChatMessage { Speaker = "Assistant:", Text = "Backend event stream is not ready yet." });
                return;
            }

            if (!await _backendHttpClient.IsHealthyAsync(_shutdown.Token))
            {
                SetBackendStatus("Disconnected", "Backend health check failed before sending.");
                ChatMessages.Add(new ChatMessage { Speaker = "Assistant:", Text = "Backend is not available yet." });
                return;
            }

            var accepted = await _backendHttpClient.StartChatAsync(message, _conversationId, _shutdown.Token);
            _conversationId = accepted.ConversationId;
            SetBackendStatus("Connected", $"Run accepted: {accepted.RunId}");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            SetBackendStatus("Error", ex.Message);
            ChatMessages.Add(new ChatMessage { Speaker = "Assistant:", Text = $"Backend error: {ex.Message}" });
            ReplaceWithSingle(FailedActions, $"Send failed: {ex.Message}");
        }
        finally
        {
            IsBusy = false;
        }
    }

    private bool CanSend()
    {
        return _isReady && !IsBusy && !string.IsNullOrWhiteSpace(DraftInput);
    }

    private async Task ApprovePermissionAsync()
    {
        if (_currentPermissionId is null)
        {
            return;
        }

        await _backendHttpClient.ApprovePermissionAsync(_currentPermissionId, _shutdown.Token);
        _currentPermissionId = null;
        ApprovePermissionCommand.RaiseCanExecuteChanged();
        RejectPermissionCommand.RaiseCanExecuteChanged();
    }

    private async Task RejectPermissionAsync()
    {
        if (_currentPermissionId is null)
        {
            return;
        }

        await _backendHttpClient.RejectPermissionAsync(_currentPermissionId, _shutdown.Token);
        _currentPermissionId = null;
        ApprovePermissionCommand.RaiseCanExecuteChanged();
        RejectPermissionCommand.RaiseCanExecuteChanged();
    }

    private bool HasPendingPermission()
    {
        return !string.IsNullOrWhiteSpace(_currentPermissionId);
    }

    private static bool HasProposedToolId(object? parameter)
    {
        return parameter is string value && !string.IsNullOrWhiteSpace(value);
    }

    private async Task ApproveProposedToolAsync(object? parameter)
    {
        if (parameter is not string toolId || string.IsNullOrWhiteSpace(toolId))
        {
            return;
        }

        await _backendHttpClient.ApproveProposedToolAsync(toolId, _shutdown.Token);
        await LoadProposedToolsAsync(_shutdown.Token);
    }

    private async Task RejectProposedToolAsync(object? parameter)
    {
        if (parameter is not string toolId || string.IsNullOrWhiteSpace(toolId))
        {
            return;
        }

        await _backendHttpClient.RejectProposedToolAsync(toolId, _shutdown.Token);
        await LoadProposedToolsAsync(_shutdown.Token);
    }

    private async Task NeedsChangesProposedToolAsync(object? parameter)
    {
        if (parameter is not string toolId || string.IsNullOrWhiteSpace(toolId))
        {
            return;
        }

        await _backendHttpClient.MarkProposedToolNeedsChangesAsync(toolId, _shutdown.Token);
        await LoadProposedToolsAsync(_shutdown.Token);
    }

    private void ApplyPlan(PlanDto plan)
    {
        GoalTitle = string.IsNullOrWhiteSpace(plan.Goal)
            ? "No active plan"
            : $"Goal: {plan.Goal}";

        PlanSteps.Clear();
        foreach (var step in plan.Steps.OrderBy(step => step.Number))
        {
            PlanSteps.Add(new PlanStep
            {
                Number = step.Number,
                Title = string.IsNullOrWhiteSpace(step.Status) ? step.Title : $"{step.Title} [{step.Status}]",
            });
        }

    }

    private void AddToolTrace(string toolName, Dictionary<string, object?> arguments, string status)
    {
        ToolTraces.Add(new ToolTraceEntry
        {
            Tool = $"Tool: {toolName}",
            Arguments = $"Args: {FormatObjectMap(arguments)}",
            Status = $"Status: {status}",
        });
        ToolTracePlaceholder = string.Empty;
    }

    private void ApplyAssistantEvent(AssistantEventDto assistantEvent)
    {
        switch (assistantEvent.Type)
        {
            case "user_message_received":
                ChatMessages.Add(new ChatMessage
                {
                    Speaker = "You:",
                    Text = GetString(assistantEvent.Data, "message"),
                });
                break;

            case "assistant_message_created":
                ChatMessages.Add(new ChatMessage
                {
                    Speaker = "Assistant:",
                    Text = GetString(assistantEvent.Data, "message"),
                });
                IsBusy = false;
                break;

            case "plan_created":
                ApplyPlanIfPresent(assistantEvent.Data);
                break;

            case "tool_selected":
                ApplyPlanIfPresent(assistantEvent.Data);
                AddToolTrace(GetString(assistantEvent.Data, "tool_name"), GetObjectMap(assistantEvent.Data, "arguments"), "selected");
                break;

            case "tool_started":
                ApplyPlanIfPresent(assistantEvent.Data);
                AddToolTrace(GetString(assistantEvent.Data, "tool_name"), GetObjectMap(assistantEvent.Data, "arguments"), "running");
                break;

            case "tool_result":
                ApplyPlanIfPresent(assistantEvent.Data);
                var resultToolName = GetString(assistantEvent.Data, "tool_name");
                var resultArguments = GetObjectMap(assistantEvent.Data, "arguments");
                var resultStatus = GetString(assistantEvent.Data, "status");
                AddToolTrace(
                    resultToolName,
                    resultArguments,
                    resultStatus);
                AddRecentToolCall(resultToolName, resultArguments, resultStatus);
                break;

            case "permission_required":
                _currentPermissionId = GetString(assistantEvent.Data, "permission_id");
                PermissionItems.Clear();
                foreach (var line in FormatPermission(assistantEvent.Data))
                {
                    PermissionItems.Add(line);
                }
                AddPermissionHistory(assistantEvent.Data, "pending");
                ApprovePermissionCommand.RaiseCanExecuteChanged();
                RejectPermissionCommand.RaiseCanExecuteChanged();
                break;

            case "permission_approved":
            case "permission_rejected":
                PermissionItems.Clear();
                PermissionItems.Add(assistantEvent.Type == "permission_approved" ? "Permission approved" : "Permission rejected");
                AddPermissionDecisionHistory(assistantEvent.Data, assistantEvent.Type == "permission_approved" ? "approved" : "rejected");
                _currentPermissionId = null;
                ApprovePermissionCommand.RaiseCanExecuteChanged();
                RejectPermissionCommand.RaiseCanExecuteChanged();
                break;

            case "proposed_tool_created":
            case "proposed_tool_approved":
            case "proposed_tool_rejected":
            case "proposed_tool_needs_changes":
                ApplyProposedToolIfPresent(assistantEvent.Data);
                break;

            case "error_occurred":
                InsertWithLimit(FailedActions, GetString(assistantEvent.Data, "message"), "No failed actions");
                break;
        }
    }

    private async Task LoadRegisteredToolsAsync(CancellationToken cancellationToken)
    {
        var tools = await _backendHttpClient.GetToolsAsync(cancellationToken);
        ImplementedTools.Clear();
        PlannedTools.Clear();
        DisabledTools.Clear();
        foreach (var tool in tools)
        {
            switch (tool.Status)
            {
                case "implemented":
                    ImplementedTools.Add(tool.Name);
                    break;
                case "disabled":
                    DisabledTools.Add(tool.Name);
                    break;
                default:
                    PlannedTools.Add(tool.Name);
                    break;
            }
        }

        if (ImplementedTools.Count == 0)
        {
            ImplementedTools.Add("No implemented tools");
        }
        if (PlannedTools.Count == 0)
        {
            PlannedTools.Add("No planned tools");
        }
        if (DisabledTools.Count == 0)
        {
            DisabledTools.Add("No disabled tools");
        }
    }

    private async Task ListenForBackendEventsAsync(CancellationToken cancellationToken)
    {
        try
        {
            await _backendWebSocketClient.ListenForEventsAsync(
                assistantEvent => Application.Current.Dispatcher.InvokeAsync(() =>
                {
                    if (!_hasReceivedEvent)
                    {
                        EventLogEntries.Clear();
                        _hasReceivedEvent = true;
                    }

                    EventLogEntries.Insert(0, FormatEventJson(assistantEvent));
                    if (assistantEvent.Type == "event_stream_connected")
                    {
                        _eventStreamReady.TrySetResult(true);
                    }
                    ApplyAssistantEvent(assistantEvent);
                    return Task.CompletedTask;
                }).Task.Unwrap(),
                cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                ReplaceWithSingle(FailedActions, $"Event stream failed: {ex.Message}");
            });
        }
    }

    private async Task WaitForBackendAsync(CancellationToken cancellationToken)
    {
        for (var attempt = 0; attempt < 30; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (await _backendHttpClient.IsHealthyAsync(cancellationToken))
            {
                return;
            }

            await Task.Delay(500, cancellationToken);
        }

        throw new TimeoutException("Backend did not become healthy within 15 seconds.");
    }

    private void SetBackendStatus(string value, string detail)
    {
        _backendStatus = value;
        _backendStatusDetail = detail;
        RefreshSettings();
    }

    private void RefreshSettings()
    {
        Settings.Clear();
        Settings.Add(new StatusItem
        {
            Label = "Backend status",
            Value = _backendStatus,
            Detail = _backendStatusDetail,
        });
        Settings.Add(new StatusItem
        {
            Label = "API key status",
            Value = "Not configured",
            Detail = "Keys stay out of WPF and will be owned by the backend.",
        });
        Settings.Add(new StatusItem
        {
            Label = "Model selection",
            Value = "Later",
            Detail = "The model picker is intentionally a disabled placeholder.",
        });
    }

    private static void ReplaceWithSingle(ObservableCollection<string> collection, string value)
    {
        collection.Clear();
        collection.Add(value);
    }

    private void AddRecentToolCall(string toolName, Dictionary<string, object?> arguments, string status)
    {
        var line = $"{toolName} [{status}] - {FormatObjectMap(arguments)}";
        InsertWithLimit(RecentToolCalls, line, "No tool calls yet");
        if (!string.Equals(status, "success", StringComparison.OrdinalIgnoreCase))
        {
            InsertWithLimit(FailedToolCalls, line, "No failed tool calls");
        }
    }

    private async Task LoadProposedToolsAsync(CancellationToken cancellationToken)
    {
        var proposedTools = await _backendHttpClient.GetProposedToolsAsync(cancellationToken);
        ProposedTools.Clear();
        foreach (var proposedTool in proposedTools)
        {
            ProposedTools.Add(proposedTool);
        }
    }

    private void AddPermissionHistory(Dictionary<string, object?> data, string status)
    {
        var permissionId = GetString(data, "permission_id");
        var preview = GetObjectMap(data, "preview");
        var action = GetPreviewString(preview, "action", GetString(data, "action_type"));
        var riskLevel = GetPreviewString(preview, "risk_level", "unknown");
        var target = GetPreviewString(preview, "target", "Unknown target");
        InsertWithLimit(PermissionHistory, $"{permissionId} [{status}] {action} -> {target} ({riskLevel})", "No permission history yet");
    }

    private void AddPermissionDecisionHistory(Dictionary<string, object?> data, string status)
    {
        var permissionId = GetString(data, "permission_id");
        InsertWithLimit(PermissionHistory, $"{permissionId} [{status}]", "No permission history yet");
    }

    private static void InsertWithLimit(ObservableCollection<string> collection, string value, string placeholder, int limit = 40)
    {
        if (collection.Count == 1 && collection[0] == placeholder)
        {
            collection.Clear();
        }

        collection.Insert(0, value);
        while (collection.Count > limit)
        {
            collection.RemoveAt(collection.Count - 1);
        }
    }

    private void ApplyPlanIfPresent(Dictionary<string, object?> data)
    {
        if (!data.TryGetValue("plan", out var value))
        {
            return;
        }

        var plan = DeserializeValue<PlanDto>(value);
        if (plan is not null)
        {
            ApplyPlan(plan);
        }
    }

    private void ApplyProposedToolIfPresent(Dictionary<string, object?> data)
    {
        if (!data.TryGetValue("tool", out var value))
        {
            return;
        }

        var proposedTool = DeserializeValue<ProposedToolDto>(value);
        if (proposedTool is not null)
        {
            UpsertProposedTool(proposedTool);
        }
    }

    private void UpsertProposedTool(ProposedToolDto proposedTool)
    {
        for (var index = 0; index < ProposedTools.Count; index++)
        {
            if (ProposedTools[index].Id == proposedTool.Id)
            {
                ProposedTools[index] = proposedTool;
                return;
            }
        }

        ProposedTools.Insert(0, proposedTool);
    }

    private static T? DeserializeValue<T>(object? value)
    {
        return value switch
        {
            JsonElement element => element.Deserialize<T>(),
            T typed => typed,
            _ => default,
        };
    }

    private static string GetString(Dictionary<string, object?> data, string key)
    {
        if (!data.TryGetValue(key, out var value) || value is null)
        {
            return string.Empty;
        }

        return value switch
        {
            JsonElement element when element.ValueKind == JsonValueKind.String => element.GetString() ?? string.Empty,
            JsonElement element => element.ToString(),
            _ => value.ToString() ?? string.Empty,
        };
    }

    private static Dictionary<string, object?> GetObjectMap(Dictionary<string, object?> data, string key)
    {
        if (!data.TryGetValue(key, out var value) || value is null)
        {
            return [];
        }

        if (value is Dictionary<string, object?> dictionary)
        {
            return dictionary;
        }

        if (value is JsonElement element && element.ValueKind == JsonValueKind.Object)
        {
            return element.EnumerateObject().ToDictionary(property => property.Name, property => (object?)property.Value.Clone());
        }

        return [];
    }

    private static IEnumerable<string> FormatPermission(Dictionary<string, object?> data)
    {
        var permissionId = GetString(data, "permission_id");
        var preview = GetObjectMap(data, "preview");
        var actionType = GetString(data, "action_type");

        yield return "Permission required";
        yield return $"Permission: {permissionId}";
        yield return $"Action: {FormatPermissionAction(actionType, preview)}";
        yield return $"Target: {FormatPermissionTarget(preview)}";
        yield return $"Preview: {FormatPermissionPreview(preview)}";
        yield return $"Risk level: {GetPreviewString(preview, "risk_level", "unknown")}";
        yield return $"What exactly will happen: {GetPreviewString(preview, "what_will_happen", "Permission is required before this action runs.")}";
        yield return $"Reason: {GetPreviewString(preview, "reason", "Permission required")}";
        yield return $"Edit: {FormatEditState(preview)}";
    }

    private static string FormatObjectMap(Dictionary<string, object?> values)
    {
        if (values.Count == 0)
        {
            return "{}";
        }

        return string.Join(", ", values.Select(pair => $"{pair.Key}={FormatObject(pair.Value)}"));
    }

    private static string FormatObject(object? value)
    {
        return value switch
        {
            null => "null",
            JsonElement element => element.ValueKind == JsonValueKind.String ? element.GetString() ?? string.Empty : element.ToString(),
            _ => value.ToString() ?? string.Empty,
        };
    }

    private static string GetPreviewString(Dictionary<string, object?> preview, string key, string fallback)
    {
        return preview.TryGetValue(key, out var value) ? FormatObject(value) : fallback;
    }

    private static string FormatPreviewValue(Dictionary<string, object?> preview, string key)
    {
        if (!preview.TryGetValue(key, out var value))
        {
            return "{}";
        }

        if (value is Dictionary<string, object?> dictionary)
        {
            return FormatObjectMap(dictionary);
        }

        if (value is JsonElement element && element.ValueKind == JsonValueKind.Object)
        {
            return FormatObjectMap(element.EnumerateObject().ToDictionary(property => property.Name, property => (object?)property.Value.Clone()));
        }

        return FormatObject(value);
    }

    private static string FormatPermissionAction(string actionType, Dictionary<string, object?> preview)
    {
        if (string.Equals(actionType, "send_message", StringComparison.OrdinalIgnoreCase)
            && preview.TryGetValue("recipient", out var recipient))
        {
            return $"Send message to {FormatObject(recipient)}";
        }

        return GetPreviewString(preview, "action", actionType);
    }

    private static string FormatPermissionTarget(Dictionary<string, object?> preview)
    {
        if (preview.TryGetValue("target", out var target))
        {
            return FormatObject(target);
        }

        if (preview.TryGetValue("recipient", out var recipient))
        {
            return FormatObject(recipient);
        }

        return "Unknown target";
    }

    private static string FormatPermissionPreview(Dictionary<string, object?> preview)
    {
        if (preview.TryGetValue("message", out var message))
        {
            return $"\"{FormatObject(message)}\"";
        }

        if (preview.TryGetValue("content", out _))
        {
            return FormatPreviewValue(preview, "content");
        }

        return FormatObjectMap(preview);
    }

    private static string FormatEditState(Dictionary<string, object?> preview)
    {
        if (!preview.TryGetValue("editable", out var value))
        {
            return "Disabled for Stage 1";
        }

        var editable = value switch
        {
            bool boolValue => boolValue,
            JsonElement element when element.ValueKind is JsonValueKind.True or JsonValueKind.False => element.GetBoolean(),
            _ => false,
        };

        return editable ? "Available" : "Disabled for Stage 1";
    }

    private static string FormatEventJson(AssistantEventDto assistantEvent)
    {
        return JsonSerializer.Serialize(assistantEvent, new JsonSerializerOptions { WriteIndented = false });
    }

    private bool SetProperty<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }

        field = value;
        OnPropertyChanged(propertyName);
        return true;
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
