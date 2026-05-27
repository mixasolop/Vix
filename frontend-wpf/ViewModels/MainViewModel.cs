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
    private static readonly Uri BackendUri = new("http://127.0.0.1:8765");

    private readonly BackendClient _backendClient = new(BackendUri);
    private readonly BackendProcessManager _backendProcessManager = new();
    private readonly CancellationTokenSource _shutdown = new();
    private string? _conversationId;
    private string _backendStatus = "Disconnected";
    private string _backendStatusDetail = "Backend has not been checked yet.";
    private string _draftInput = string.Empty;
    private string _goalTitle = "No active plan";
    private string _toolTracePlaceholder = "No tool calls yet";
    private bool _isBusy;
    private bool _hasReceivedEvent;

    public MainViewModel()
    {
        SendCommand = new AsyncRelayCommand(SendAsync, CanSend);

        PermissionItems.Add("No pending permissions");
        RegisteredTools.Add("Backend not connected");
        EventLogEntries.Add("No events received");
        FailedActions.Add("No failed actions");
        FutureProposedTools.Add("window_focus");
        FutureProposedTools.Add("clipboard_read");
        FutureProposedTools.Add("clipboard_write");
        FutureProposedTools.Add("screenshot_region");

        RefreshSettings();
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public AsyncRelayCommand SendCommand { get; }

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

    public ObservableCollection<string> RegisteredTools { get; } = [];

    public ObservableCollection<string> EventLogEntries { get; } = [];

    public ObservableCollection<string> FailedActions { get; } = [];

    public ObservableCollection<string> FutureProposedTools { get; } = [];

    public ObservableCollection<StatusItem> Settings { get; } = [];

    public async Task InitializeAsync()
    {
        SetBackendStatus("Starting", "Checking local backend health.");

        try
        {
            if (!await _backendClient.IsHealthyAsync(_shutdown.Token))
            {
                var backendDirectory = _backendProcessManager.LocateBackendDirectory();
                _backendProcessManager.StartBackend(backendDirectory);
                await WaitForBackendAsync(_shutdown.Token);
            }

            SetBackendStatus("Connected", "Backend is available at http://127.0.0.1:8765.");
            await LoadRegisteredToolsAsync(_shutdown.Token);
            _ = Task.Run(() => ListenForBackendEventsAsync(_shutdown.Token));
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            SetBackendStatus("Error", ex.Message);
            ReplaceWithSingle(FailedActions, $"Backend startup failed: {ex.Message}");
        }
    }

    public void Dispose()
    {
        _shutdown.Cancel();
        _shutdown.Dispose();
        _backendClient.Dispose();
        _backendProcessManager.Dispose();
    }

    private async Task SendAsync()
    {
        var message = DraftInput.Trim();
        if (message.Length == 0)
        {
            return;
        }

        ChatMessages.Add(new ChatMessage { Speaker = "You:", Text = message });
        DraftInput = string.Empty;
        IsBusy = true;

        try
        {
            if (!await _backendClient.IsHealthyAsync(_shutdown.Token))
            {
                SetBackendStatus("Disconnected", "Backend health check failed before sending.");
                ChatMessages.Add(new ChatMessage { Speaker = "Assistant:", Text = "Backend is not available yet." });
                return;
            }

            var response = await _backendClient.SendChatAsync(message, _conversationId, _shutdown.Token);
            _conversationId = response.ConversationId;
            ApplyChatResponse(response);
            SetBackendStatus("Connected", "Last chat request completed.");
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
        return !IsBusy && !string.IsNullOrWhiteSpace(DraftInput);
    }

    private void ApplyChatResponse(ChatResponseDto response)
    {
        ChatMessages.Add(new ChatMessage { Speaker = "Assistant:", Text = response.AssistantMessage });

        GoalTitle = string.IsNullOrWhiteSpace(response.Plan.Goal)
            ? "No active plan"
            : $"Goal: {response.Plan.Goal}";

        PlanSteps.Clear();
        foreach (var step in response.Plan.Steps.OrderBy(step => step.Number))
        {
            PlanSteps.Add(new PlanStep
            {
                Number = step.Number,
                Title = string.IsNullOrWhiteSpace(step.Status) ? step.Title : $"{step.Title} [{step.Status}]",
            });
        }

        ToolTraces.Clear();
        foreach (var call in response.ToolCalls)
        {
            ToolTraces.Add(new ToolTraceEntry
            {
                Tool = $"Tool: {call.Tool}",
                Arguments = $"Args: {FormatObjectMap(call.Arguments)}",
                Status = $"Status: {call.Status}",
            });
        }

        ToolTracePlaceholder = ToolTraces.Count == 0 ? "No tool calls yet" : string.Empty;

        PermissionItems.Clear();
        if (response.Permissions.Count == 0)
        {
            PermissionItems.Add("No pending permissions");
        }
        else
        {
            foreach (var permission in response.Permissions)
            {
                PermissionItems.Add($"{permission.PermissionId}: {permission.Tool} - {permission.Reason}");
            }
        }
    }

    private async Task LoadRegisteredToolsAsync(CancellationToken cancellationToken)
    {
        var toolNames = await _backendClient.GetToolNamesAsync(cancellationToken);
        RegisteredTools.Clear();
        foreach (var toolName in toolNames)
        {
            RegisteredTools.Add(toolName);
        }

        if (RegisteredTools.Count == 0)
        {
            RegisteredTools.Add("No tools registered");
        }
    }

    private async Task ListenForBackendEventsAsync(CancellationToken cancellationToken)
    {
        try
        {
            await _backendClient.ListenForEventsAsync(
                eventType => Application.Current.Dispatcher.InvokeAsync(() =>
                {
                    if (!_hasReceivedEvent)
                    {
                        EventLogEntries.Clear();
                        _hasReceivedEvent = true;
                    }

                    EventLogEntries.Insert(0, eventType);
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
            if (await _backendClient.IsHealthyAsync(cancellationToken))
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
