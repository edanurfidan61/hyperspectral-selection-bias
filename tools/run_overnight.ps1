# revision-1 overnight chain - SUPERVISOR loop (orchestrator logic: tools\run_overnight.py).
# ASCII only on purpose: Windows PowerShell 5.1 reads BOM-less .ps1 files as ANSI.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_overnight.ps1            # supervised chain
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_overnight.ps1 --dry-run  # one pass, no loop
#
# Orchestrator exit codes:
#   0 all steps DONE        -> supervisor exits
#   2 lock held elsewhere   -> supervisor exits (another chain is running)
#   3 STOP / STOP_AFTER     -> supervisor exits
#   4 wait-until reached    -> supervisor exits (FAILED/SKIPPED steps remain; see status json)
#   5 restart requested     -> restart after 5 s (CONTROL restart_orchestrator; not a crash)
#   other (crash, killed, unexpected exit) -> wait 60 s, restart; at most 5 crash restarts
#         per rolling hour (beyond that the supervisor waits until the window frees up).
# Every restart is a NEW python process, so code fixes written to the repo take effect.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$py   = Join-Path $root '.venv\Scripts\python.exe'
$orch = Join-Path $root 'tools\run_overnight.py'
if (-not (Test-Path -LiteralPath $py)) { Write-Error ".venv python not found: $py"; exit 9 }
$env:PYTHONIOENCODING = 'utf-8'

# Single pass (no supervision) for dry-run / help. (--simulate IS supervised, so the
# supervisor itself can be exercised in a sandbox: --simulate --logs-dir <dir>.)
if ($args -contains '--dry-run' -or $args -contains '-h' -or $args -contains '--help') {
    & $py $orch @args
    exit $LASTEXITCODE
}

# Log next to the chain's logs (honours --logs-dir if given)
$logs = Join-Path $root 'outputs_rev1\logs'
$i = [Array]::IndexOf($args, '--logs-dir')
if ($i -ge 0 -and $i + 1 -lt $args.Count) { $logs = $args[$i + 1] }
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$slog = Join-Path $logs 'overnight_supervisor.log'
function Write-SupLog([string]$msg) {
    $line = '{0:yyyy-MM-dd HH:mm:ss} | supervisor pid {1} | {2}' -f (Get-Date), $PID, $msg
    Add-Content -LiteralPath $slog -Value $line -Encoding UTF8
    Write-Host $line
}

$maxPerHour = 5
$crashTimes = New-Object System.Collections.Generic.List[datetime]
$restarts = 0
$lastExit = ''
Write-SupLog "start | args: $($args -join ' ')"
while ($true) {
    $env:REV1_SUPERVISOR_RESTARTS = "$restarts"
    $env:REV1_SUPERVISOR_LAST_EXIT = "$lastExit"
    $t0 = Get-Date
    & $py $orch @args
    $code = $LASTEXITCODE
    $lastExit = "$code"
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    # NOTE: if/elseif, not switch - 'continue' inside a PowerShell switch does not reach the while loop
    if ($code -eq 0) { Write-SupLog "orchestrator exit 0 (all steps DONE) after $mins min -> supervisor exits"; exit 0 }
    elseif ($code -eq 2) { Write-SupLog "orchestrator exit 2 (lock held by another chain) -> supervisor exits"; exit 2 }
    elseif ($code -eq 3) { Write-SupLog "orchestrator exit 3 (STOP file) -> supervisor exits"; exit 3 }
    elseif ($code -eq 4) { Write-SupLog "orchestrator exit 4 (wait-until reached; FAILED/SKIPPED remain) -> supervisor exits"; exit 4 }
    elseif ($code -eq 5) {
        $restarts++
        Write-SupLog "orchestrator exit 5 (restart requested via CONTROL) -> restart #$restarts in 5 s"
        Start-Sleep -Seconds 5
        continue
    }
    # Abnormal end (crash / killed / unexpected exit code)
    if (Test-Path -LiteralPath (Join-Path $logs 'overnight.STOP')) { Write-SupLog "exit $code but STOP file present -> exit"; exit 3 }
    $now = Get-Date
    $cutoff = $now.AddHours(-1)
    $crashTimes.RemoveAll([Predicate[datetime]]{ param($t) $t -lt $cutoff }) | Out-Null
    if ($crashTimes.Count -ge $maxPerHour) {
        $waitUntil = $crashTimes[0].AddHours(1)
        $secs = [math]::Max(60, [int]($waitUntil - $now).TotalSeconds)
        Write-SupLog "abnormal exit $code after $mins min; $maxPerHour crash restarts in the last hour -> waiting $secs s"
        Start-Sleep -Seconds $secs
        $crashTimes.RemoveAll([Predicate[datetime]]{ param($t) $t -lt (Get-Date).AddHours(-1) }) | Out-Null
    } else {
        Write-SupLog "abnormal exit $code after $mins min -> restart in 60 s ($($crashTimes.Count + 1)/$maxPerHour this hour)"
        Start-Sleep -Seconds 60
    }
    if (Test-Path -LiteralPath (Join-Path $logs 'overnight.STOP')) { Write-SupLog "STOP file present -> exit"; exit 3 }
    $crashTimes.Add((Get-Date))
    $restarts++
}
