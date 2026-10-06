$ErrorActionPreference = 'Stop'
try {
    $taskConfigPath = Join-Path $PSScriptRoot 'local-runtime.json'
    $taskConfig = if (Test-Path -LiteralPath $taskConfigPath) { Get-Content -LiteralPath $taskConfigPath -Raw | ConvertFrom-Json } else { $null }
    $taskPython = if ($taskConfig -and $taskConfig.python_executable) { $taskConfig.python_executable } else { 'python' }
    if ([IO.Path]::GetFileName($taskPython) -eq 'pythonw.exe') { $taskPython = Join-Path (Split-Path $taskPython) 'python.exe' }
    Write-Host 'Opening ECG-PPG_Neotex. Please wait...'
    & $taskPython (Join-Path $PSScriptRoot 'neotex_desktop.py')
    exit $LASTEXITCODE
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
}