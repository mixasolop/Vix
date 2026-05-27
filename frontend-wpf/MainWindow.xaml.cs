using System.Windows;
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
        Closed += (_, _) => ViewModel.Dispose();
    }
}
