// Portable installer: download split ZIP, verify the complete archive, extract without overwriting an installation.
// Templates are substituted by package_installer.py after the ZIP is finalized.
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("RuVoice Setup")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]
[assembly: System.Reflection.AssemblyFileVersion("1.0.0.0")]

static class Installer
{
    const string Archive = "RuVoice-1.0.0-Windows-x64.zip";
    const string ExpectedHash = "__ARCHIVE_SHA256__";
    const int Parts = __PARTS__;
    const string DownloadRoot = "https://github.com/berem-git/ruVoice/releases/download/v1.0.0/";

    [STAThread]
    static int Main(string[] args)
    {
        // Reproducible verification of the same extraction code without opening a window.
        if (args.Length == 3 && args[0] == "--verify-and-extract")
        {
            try { ExtractVerified(args[1], args[2]); return 0; }
            catch (Exception e) { File.WriteAllText(args[1] + ".error.txt", e.ToString()); return 1; }
        }
        Application.EnableVisualStyles();
        var form = new Form { Text = "RuVoice 1.0.0 — установка", Width = 650, Height = 290,
            StartPosition = FormStartPosition.CenterScreen, FormBorderStyle = FormBorderStyle.FixedDialog,
            MaximizeBox = false };
        var text = new Label { Left = 20, Top = 15, Width = 600, Height = 65,
            Text = "Будут скачаны программа, встроенный Python и CUDA. Модели скачиваются позже из программы.\nКачество озвучки экспериментальное: сильный акцент пока не исправлен." };
        var path = new TextBox { Left = 20, Top = 90, Width = 490,
            Text = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "RuVoice") };
        var browse = new Button { Left = 520, Top = 88, Width = 95, Text = "Папка…" };
        browse.Click += (s, e) => { using (var d = new FolderBrowserDialog()) {
            d.Description = "Куда распаковать RuVoice"; if (d.ShowDialog() == DialogResult.OK) path.Text = d.SelectedPath; } };
        var status = new Label { Left = 20, Top = 130, Width = 600, Height = 45,
            Text = "Нужны интернет и около 15 ГиБ свободного места на время установки." };
        var progress = new ProgressBar { Left = 20, Top = 180, Width = 590, Height = 16 };
        var start = new Button { Left = 390, Top = 210, Width = 220, Text = "Скачать и установить" };
        bool busy = false;
        form.FormClosing += (s, e) => { if (busy) e.Cancel = true; };
        Action<string> update = message => form.BeginInvoke((Action)(() => status.Text = message));
        start.Click += async (s, e) => {
            string target = Path.GetFullPath(path.Text);
            string installed = Path.Combine(target, "RuVoice-1.0.0");
            if (Directory.Exists(installed)) { MessageBox.Show("Эта папка уже существует. Выберите другую папку — текущую установку перезаписывать не будем."); return; }
            busy = true; start.Enabled = false; browse.Enabled = false; path.Enabled = false;
            string cache = Path.Combine(Path.GetTempPath(), "RuVoice-setup-" + Guid.NewGuid().ToString("N"));
            try {
                await Task.Run(() => {
                    Directory.CreateDirectory(cache);
                    string combined = Path.Combine(cache, Archive);
                    using (var output = new FileStream(combined, FileMode.CreateNew))
                    for (int i = 1; i <= Parts; i++) {
                        string name = Parts == 1 ? Archive : Archive + "." + i.ToString("000");
                        string local = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, name);
                        if (!File.Exists(local)) {
                            update("Скачивание части " + i + "/" + Parts + "…");
                            local = Path.Combine(cache, name);
                            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
                            using (var client = new WebClient()) {
                                client.Headers[HttpRequestHeader.UserAgent] = "RuVoice-Setup/1.0.0";
                                client.DownloadProgressChanged += (sender, ev) => form.BeginInvoke((Action)(() => {
                                    progress.Value = Math.Max(0, Math.Min(100, ev.ProgressPercentage));
                                    status.Text = "Часть " + i + "/" + Parts + ": " + (ev.BytesReceived / 1048576) + " МиБ";
                                }));
                                client.DownloadFileTaskAsync(new Uri(DownloadRoot + name), local).GetAwaiter().GetResult();
                            }
                        }
                        update("Объединение части " + i + "/" + Parts + "…");
                        using (var input = File.OpenRead(local)) input.CopyTo(output);
                    }
                    update("Проверка SHA-256 и распаковка…");
                    ExtractVerified(combined, target);
                    update("Установка NVIDIA CUDA из официального репозитория PyTorch…");
                    var uv = new ProcessStartInfo(Path.Combine(installed, @"tools\uv\uv.exe"),
                        "pip install --no-deps --link-mode copy --target \"" + Path.Combine(installed, @"runtime\accel\cuda") +
                        "\" --python \"" + Path.Combine(installed, @"runtime\python\python.exe") +
                        "\" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchaudio==2.11.0+cu128") {
                        WorkingDirectory = installed, UseShellExecute = false, CreateNoWindow = true,
                        RedirectStandardOutput = true, RedirectStandardError = true };
                    uv.EnvironmentVariables["UV_CACHE_DIR"] = Path.Combine(installed, @"cache\uv");
                    using (var process = Process.Start(uv)) {
                        var stdout = process.StandardOutput.ReadToEndAsync();
                        var stderr = process.StandardError.ReadToEndAsync();
                        process.WaitForExit();
                        if (process.ExitCode != 0) throw new IOException("Не удалось скачать CUDA. Запустите «Установить CUDA.cmd» из папки программы.\n" + stderr.Result);
                    }
                    // Delete only installer-owned temporary files, never the chosen destination.
                    foreach (string f in Directory.GetFiles(cache)) File.Delete(f);
                    Directory.Delete(cache);
                });
                status.Text = "Готово. Запускаю RuVoice. Модели доступны в «Система и ускорение».";
                Process.Start(new ProcessStartInfo(Path.Combine(installed, "RuVoice.exe")) { WorkingDirectory = installed, UseShellExecute = true });
                busy = false; form.Close();
            } catch (Exception error) {
                busy = false; start.Enabled = true; browse.Enabled = true; path.Enabled = true;
                status.Text = "Установка не завершена. Файлы игры не изменялись.";
                MessageBox.Show(error.Message + "\n\nВременные файлы: " + cache +
                    "\nЕсли репозиторий закрыт, скачайте части архива с доступной вам страницы релиза и положите рядом с установщиком.", "RuVoice", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        };
        form.Controls.AddRange(new Control[] { text, path, browse, status, progress, start });
        Application.Run(form); return 0;
    }

    static void ExtractVerified(string archive, string destination)
    {
        string hash;
        using (var sha = SHA256.Create()) using (var input = File.OpenRead(archive))
            hash = BitConverter.ToString(sha.ComputeHash(input)).Replace("-", "").ToLowerInvariant();
        if (hash != ExpectedHash) throw new InvalidDataException("Контрольная сумма архива не совпала. Скачайте части заново.");
        destination = Path.GetFullPath(destination);
        if (Directory.Exists(Path.Combine(destination, "RuVoice-1.0.0")))
            throw new IOException("Папка установки уже существует; выберите другую папку.");
        using (var zip = ZipFile.OpenRead(archive)) {
            string prefix = destination.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
            foreach (var entry in zip.Entries) {
                string final = Path.GetFullPath(Path.Combine(destination, entry.FullName));
                if (!final.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("Недопустимый путь в архиве.");
            }
        }
        Directory.CreateDirectory(destination);
        ZipFile.ExtractToDirectory(archive, destination);
    }
}
