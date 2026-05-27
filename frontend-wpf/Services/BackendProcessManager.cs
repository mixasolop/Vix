using System.Diagnostics;
using System.IO;

namespace DesktopAssistant.Frontend.Services;

public sealed class BackendProcessManager : IDisposable
{
    private const string BackendRelativePath = "backend";
    private Process? _process;

    public DirectoryInfo LocateBackendDirectory()
    {
        foreach (var startPath in GetSearchRoots())
        {
            var current = new DirectoryInfo(startPath);
            while (current is not null)
            {
                var candidate = new DirectoryInfo(Path.Combine(current.FullName, BackendRelativePath));
                if (File.Exists(Path.Combine(candidate.FullName, "app", "main.py")))
                {
                    return candidate;
                }

                current = current.Parent;
            }
        }

        throw new DirectoryNotFoundException("Could not find backend/app/main.py from the WPF runtime directory.");
    }

    public Process StartBackend(DirectoryInfo backendDirectory)
    {
        if (_process is { HasExited: false })
        {
            return _process;
        }

        var python = ResolvePythonCommand(backendDirectory);
        var startInfo = new ProcessStartInfo
        {
            FileName = python.FileName,
            Arguments = $"{python.PrefixArguments} -m uvicorn app.main:app --host 127.0.0.1 --port 8765".Trim(),
            WorkingDirectory = backendDirectory.FullName,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        _process = Process.Start(startInfo) ?? throw new InvalidOperationException("Failed to start backend process.");
        return _process;
    }

    private static IEnumerable<string> GetSearchRoots()
    {
        yield return AppContext.BaseDirectory;
        yield return Environment.CurrentDirectory;
    }

    private static PythonCommand ResolvePythonCommand(DirectoryInfo backendDirectory)
    {
        var venvPython = Path.Combine(backendDirectory.FullName, ".venv", "Scripts", "python.exe");
        if (File.Exists(venvPython))
        {
            return new PythonCommand(venvPython, string.Empty);
        }

        if (PythonLauncherVersionExists("-3.12"))
        {
            return new PythonCommand("py", "-3.12");
        }

        return new PythonCommand("python", string.Empty);
    }

    private static bool CommandExists(string command)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "where",
                Arguments = command,
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            });

            process?.WaitForExit(2000);
            return process?.ExitCode == 0;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    private static bool PythonLauncherVersionExists(string version)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "py",
                Arguments = $"{version} --version",
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            });

            process?.WaitForExit(2000);
            return process?.ExitCode == 0;
        }
        catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            return false;
        }
    }

    public void Dispose()
    {
        if (_process is null)
        {
            return;
        }

        try
        {
            if (!_process.HasExited)
            {
                _process.Kill(entireProcessTree: true);
            }
        }
        catch (InvalidOperationException)
        {
        }
        finally
        {
            _process.Dispose();
            _process = null;
        }
    }

    private sealed record PythonCommand(string FileName, string PrefixArguments);
}
