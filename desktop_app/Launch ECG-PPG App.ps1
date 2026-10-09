$ErrorActionPreference = 'Stop'
try {
    $taskConfigPath = Join-Path $PSScriptRoot 'local-runtime.json'
    $taskConfig = if (Test-Path -LiteralPath $taskConfigPath) { Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json } else { $null }
    $taskPython = if ($taskConfig -and $taskConfig.python_executable) { $taskConfig.python_executable } else { $null }
    if ($taskPython -and [IO.Path]::GetFileName($taskPython) -eq 'pythonw.exe') { $taskPython = Join-Path (Split-Path $taskPython) 'python.exe' }
    if (-not $taskPython) {
        $taskCandidate = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($taskCandidate -and $taskCandidate.Source -notlike '*WindowsApps*') { $taskPython = $taskCandidate.Source }
    }
    if (-not $taskPython -or -not (Test-Path -LiteralPath $taskPython)) { throw 'Python was not found. Install Python and requirements.txt, or configure python_executable in local-runtime.json.' }
    Write-Host 'Opening ECG-PPG_Neotex. Loading data can take a moment...'
    & $taskPython (Join-Path $PSScriptRoot 'neotex_desktop.py')
    exit $LASTEXITCODE
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
}
