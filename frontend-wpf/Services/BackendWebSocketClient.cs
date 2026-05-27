using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using DesktopAssistant.Frontend.Dtos;

namespace DesktopAssistant.Frontend.Services;

public sealed class BackendWebSocketClient
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private readonly Uri _eventsUri;

    public BackendWebSocketClient(Uri baseUri)
    {
        var builder = new UriBuilder(baseUri)
        {
            Scheme = baseUri.Scheme == "https" ? "wss" : "ws",
            Path = "/ws/events",
        };
        _eventsUri = builder.Uri;
    }

    public async Task ListenForEventsAsync(Func<AssistantEventDto, Task> onEvent, CancellationToken cancellationToken)
    {
        using var socket = new ClientWebSocket();
        await socket.ConnectAsync(_eventsUri, cancellationToken);

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
}
