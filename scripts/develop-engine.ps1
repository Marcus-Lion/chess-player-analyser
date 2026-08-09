param(
    [switch]$KillLocks
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$maturin = Join-Path $repoRoot '.venv\Scripts\maturin.exe'
$manifest = Join-Path $repoRoot 'engine\Cargo.toml'

$lockingProcesses = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -eq 'python.exe' -and (
            $_.CommandLine -match 'debugpy' -or
            $_.CommandLine -match 'multiprocessing-fork' -or
            $_.CommandLine -match 'spawn_main' -or
            $_.CommandLine -match [regex]::Escape($venvPython)
        )
    } |
    Sort-Object ProcessId -Unique

if ($lockingProcesses) {
    if (-not $KillLocks) {
        Write-Host 'Detected Python processes that may lock chess_engine.cp314-win_amd64.pyd:'
        $lockingProcesses | Select-Object ProcessId, ParentProcessId, CommandLine | Format-Table -Wrap -AutoSize
        Write-Host ''
        Write-Host 'Rerun with -KillLocks to stop only those processes, or close the matching PyCharm/debug session first.'
        exit 1
    }

    $lockingProcesses | ForEach-Object {
        try {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
        } catch {
            if ($_.Exception.Message -notmatch 'cannot find a process') {
                throw
            }
        }
    }
}

& $maturin develop --release -m $manifest
