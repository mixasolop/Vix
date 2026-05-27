using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using DesktopAssistant.Frontend.Dtos;

namespace DesktopAssistant.Frontend.Services;

public sealed class BackendHttpClient : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private readonly HttpClient _httpClient;

    public BackendHttpClient(Uri baseUri)
    {
        _httpClient = new HttpClient
        {
            BaseAddress = baseUri,
            Timeout = TimeSpan.FromSeconds(10),
        };
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

    public async Task<ChatAcceptedDto> StartChatAsync(string message, string? conversationId, CancellationToken cancellationToken)
    {
        var request = new ChatRequestDto
        {
            Message = message,
            ConversationId = conversationId,
        };

        using var response = await _httpClient.PostAsJsonAsync("/chat", request, JsonOptions, cancellationToken);
        response.EnsureSuccessStatusCode();

        var accepted = await response.Content.ReadFromJsonAsync<ChatAcceptedDto>(JsonOptions, cancellationToken);
        return accepted ?? throw new InvalidOperationException("Backend returned an empty chat acceptance response.");
    }

    public async Task<IReadOnlyList<ToolDefinitionDto>> GetToolsAsync(CancellationToken cancellationToken)
    {
        using var response = await _httpClient.GetAsync("/tools", cancellationToken);
        response.EnsureSuccessStatusCode();

        var toolsResponse = await response.Content.ReadFromJsonAsync<ToolListResponseDto>(JsonOptions, cancellationToken);
        return toolsResponse?.Tools ?? [];
    }

    public async Task<IReadOnlyList<ProposedToolDto>> GetProposedToolsAsync(CancellationToken cancellationToken)
    {
        using var response = await _httpClient.GetAsync("/proposed-tools", cancellationToken);
        response.EnsureSuccessStatusCode();

        var toolsResponse = await response.Content.ReadFromJsonAsync<ProposedToolListResponseDto>(JsonOptions, cancellationToken);
        return toolsResponse?.Tools ?? [];
    }

    public async Task ApprovePermissionAsync(string permissionId, CancellationToken cancellationToken)
    {
        using var response = await _httpClient.PostAsync($"/permissions/{permissionId}/approve", content: null, cancellationToken);
        response.EnsureSuccessStatusCode();
    }

    public async Task RejectPermissionAsync(string permissionId, CancellationToken cancellationToken)
    {
        using var response = await _httpClient.PostAsync($"/permissions/{permissionId}/reject", content: null, cancellationToken);
        response.EnsureSuccessStatusCode();
    }

    public async Task ApproveProposedToolAsync(string toolId, CancellationToken cancellationToken)
    {
        using var response = await _httpClient.PostAsync($"/proposed-tools/{toolId}/approve", content: null, cancellationToken);
        response.EnsureSuccessStatusCode();
    }

    public async Task RejectProposedToolAsync(string toolId, CancellationToken cancellationToken)
    {
        using var response = await _httpClient.PostAsync($"/proposed-tools/{toolId}/reject", content: null, cancellationToken);
        response.EnsureSuccessStatusCode();
    }

    public async Task MarkProposedToolNeedsChangesAsync(string toolId, CancellationToken cancellationToken)
    {
        using var response = await _httpClient.PostAsync($"/proposed-tools/{toolId}/needs-changes", content: null, cancellationToken);
        response.EnsureSuccessStatusCode();
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}
