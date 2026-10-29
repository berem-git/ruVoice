# Собирает G:\RuVoice\RuVoice.exe встроенным в Windows компилятором C# (.NET Framework 4).
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$csc = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
& $csc /nologo /target:winexe /optimize+ /codepage:65001 `
    /win32icon:"$here\ruvoice.ico" `
    /reference:System.Windows.Forms.dll `
    /out:"$root\RuVoice.exe" "$here\RuVoice.cs"
if ($LASTEXITCODE -eq 0) { Write-Output "Built: $root\RuVoice.exe" }
