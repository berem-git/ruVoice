// Запускатель RuVoice: задаёт окружение (все пути относительно папки программы, поэтому её
// можно переносить на другой диск или компьютер) и стартует runtime\python\pythonw.exe -m ruvoice.main.
// Сборка: launcher\build.ps1
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("RuVoice")]
[assembly: System.Reflection.AssemblyProduct("RuVoice")]
[assembly: System.Reflection.AssemblyDescription("Русская озвучка для игр")]
[assembly: System.Reflection.AssemblyVersion("1.1.0.0")]
[assembly: System.Reflection.AssemblyFileVersion("1.1.0.0")]

static class Launcher
{
    [STAThread]
    static int Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
        string pythonw = Path.Combine(root, @"runtime\python\pythonw.exe");
        if (!File.Exists(pythonw))
        {
            MessageBox.Show("Не найден встроенный Python:\n" + pythonw +
                "\n\nRuVoice.exe должен лежать в папке программы (рядом с папками runtime и ruvoice).",
                "RuVoice", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }

        var psi = new ProcessStartInfo(pythonw, "-m ruvoice.main " + string.Join(" ", args.Select(Quote)))
        {
            WorkingDirectory = root,
            UseShellExecute = false,
        };
        var env = psi.EnvironmentVariables;
        env["HF_HOME"] = Path.Combine(root, @"models\huggingface");
        env["TORCH_HOME"] = Path.Combine(root, @"models\torch");
        env["UV_CACHE_DIR"] = Path.Combine(root, @"cache\uv");
        env["PYTHONUTF8"] = "1";
        env["HF_HUB_DISABLE_XET"] = "1";  // протокол Xet качает в разы медленнее обычного HTTP
        env["PYTHONPATH"] = root;
        env["PYTHONNOUSERSITE"] = "1";  // не подхватывать пакеты из Python, установленного в системе
        env.Remove("PYTHONHOME");

        // Самая новая установленная версия Wwise
        string wwiseDir = Path.Combine(root, @"tools\Wwise");
        if (Directory.Exists(wwiseDir))
        {
            string newest = Directory.GetDirectories(wwiseDir, "Wwise_*").OrderByDescending(d => d).FirstOrDefault();
            if (newest != null)
            {
                env["WWISEROOT"] = newest;
                env["WWISE_CONSOLE"] = Path.Combine(newest, @"Authoring\x64\Release\bin\WwiseConsole.exe");
            }
        }

        string[] extraPath = { @"tools\ffmpeg\bin", @"tools\vgmstream", @"tools\quickbms", @"tools\uv" };
        env["PATH"] = string.Join(";", extraPath.Select(p => Path.Combine(root, p))) + ";" + env["PATH"];

        // Сторож: ждём, пока программа закроется. Если она умерла посреди работы (остался data\job.json и
        // код выхода не 0) — перезапускаем, и она сама продолжит прерванную озвучку. Не больше 3 раз за час.
        // Код выхода пишем в журнал всегда: так видно, почему процесс исчез.
        string log = Path.Combine(root, @"data\ruvoice.log");
        string job = Path.Combine(root, @"data\job.json");
        var restarts = new System.Collections.Generic.List<DateTime>();
        while (true)
        {
            Process p;
            try
            {
                p = Process.Start(psi);
            }
            catch (Exception e)
            {
                MessageBox.Show("Не удалось запустить RuVoice:\n" + e.Message, "RuVoice",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
            p.WaitForExit();
            int code = p.ExitCode;
            bool interrupted = code != 0 && File.Exists(job);
            Log(log, string.Format("запускатель: программа (pid {0}) завершилась с кодом {1} (0x{1:X8}){2}",
                p.Id, code, interrupted ? " посреди работы" : ""));
            if (!interrupted)
                return 0;
            restarts.RemoveAll(t => (DateTime.Now - t).TotalMinutes > 60);
            if (restarts.Count >= 3)
            {
                Log(log, "запускатель: за час уже 3 перезапуска — больше не перезапускаю");
                return 1;
            }
            restarts.Add(DateTime.Now);
            System.Threading.Thread.Sleep(15000);  // дать системе освободить видеопамять
            Log(log, "запускатель: перезапуск — программа продолжит прерванную работу");
        }
    }

    static void Log(string path, string line)
    {
        try
        {
            File.AppendAllText(path, DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss ") + line + Environment.NewLine,
                new System.Text.UTF8Encoding(false));
        }
        catch { }
    }

    static string Quote(string a)
    {
        return a.Contains(" ") ? "\"" + a + "\"" : a;
    }
}
