# Собирает запускатель встроенным в Windows компилятором C# (.NET Framework 4).
param([string]$Output)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
if (-not $Output) { $Output = Join-Path $root 'RuVoice.exe' }
$csc = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
& $csc /nologo /target:winexe /optimize+ /codepage:65001 `
    /win32icon:"$here\ruvoice.ico" `
    /reference:System.Windows.Forms.dll `
    /out:"$Output" "$here\RuVoice.cs"
if ($LASTEXITCODE -ne 0) { throw 'Launcher build failed' }
Write-Output "Built: $Output"
