using System.Net.Http;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using DesktopAssistant.Frontend.Dtos;

namespace DesktopAssistant.Frontend.Services;

public sealed class BackendClient : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private readonly HttpClient _httpClient;
    private readonly Uri _baseUri;

    public BackendClient(Uri baseUri)
    {
        _baseUri = baseUri;
        _httpClient = new HttpClient
        {
            BaseAddress = baseUri,
            Timeout = TimeSpan.FromSeconds(10),
        };
    }

    public Uri EventsUri
    {
        get
        {
            var builder = new UriBuilder(_baseUri)
            {
                Scheme = _baseUri.Scheme == "https" ? "wss" : "ws",
                Path = "/ws/events",
            };
            return builder.Uri;
        }
    }

    public async Task<bool> IsHealthyAsync(CancellationToken cancellationToken)
    {
        try
        {
            using var response = await _httpClient.GetAsync("/health", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch (HttpRequestException)
        {
            return false;
        }
        catch (TaskCanceledException)
        {
            return false;
        }
    }

    public async Task<ChatResponseDto> SendChatAsync(string message, string? conversationId, CancellationToken cancellationToken)
    {
        var request = new ChatRequestDto
        {
            Message = message,
            ConversationId = conversationId,
        };

        using var response = await _httpClient.PostAsJsonAsync("/chat", request, JsonOptions, cancellationToken);
        response.EnsureSuccessStatusCode();

        var chatResponse = await response.Content.ReadFromJsonAsync<ChatResponseDto>(JsonOptions, cancellationToken);
        return chatResponse ?? throw new InvalidOperationException("Backend returned an empty chat response.");
    }

    public async Task<IReadOnlyList<ToolDefinitionDto>> GetToolsAsync(CancellationToken cancellationToken)
    {
        using var response = await _httpClient.GetAsync("/tools", cancellationToken);
        response.EnsureSuccessStatusCode();

        var toolsResponse = await response.Content.ReadFromJsonAsync<ToolListResponseDto>(JsonOptions, cancellationToken);
        return toolsResponse?.Tools ?? [];
    }

    public async Task ListenForEventsAsync(Func<AssistantEventDto, Task> onEvent, CancellationToken cancellationToken)
    {
        using var socket = new ClientWebSocket();
        await socket.ConnectAsync(EventsUri, cancellationToken);

        var buffer = new byte[8192];
        while (!cancellationToken.IsCancellationRequested && socket.State == WebSocketState.Open)
        {
            var message = new StringBuilder();
            WebSocketReceiveResult result;
            do
            {
                result = await socket.ReceiveAsync(buffer, cancellationToken);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    return;
                }

                message.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
            }
            while (!result.EndOfMessage);

            var eventDto = JsonSerializer.Deserialize<AssistantEventDto>(message.ToString(), JsonOptions);
            if (!string.IsNullOrWhiteSpace(eventDto?.Type))
            {
                await onEvent(eventDto);
            }
        }
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}
