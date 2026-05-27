using System.Windows;
using System.Collections.Specialized;
using DesktopAssistant.Frontend.ViewModels;

namespace DesktopAssistant.Frontend;

public partial class MainWindow : Window
{
    public MainViewModel ViewModel { get; } = new();

    public MainWindow()
    {
        InitializeComponent();
        DataContext = ViewModel;
        Loaded += async (_, _) => await ViewModel.InitializeAsync();
        ViewModel.ChatMessages.CollectionChanged += ChatMessagesChanged;
        Closed += (_, _) => ViewModel.Dispose();
    }

    private void ChatMessagesChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        Dispatcher.BeginInvoke(() => ChatScrollViewer.ScrollToEnd());
    }
}
