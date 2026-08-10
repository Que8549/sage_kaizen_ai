<#
.SYNOPSIS
    Move PostgreSQL's pg_wal off the HDD onto NVMe, via a directory junction.

.DESCRIPTION
    MUST BE RUN AS ADMINISTRATOR. Stopping the PostgreSQL service and writing
    inside PGDATA both require elevation; a normal shell fails with
    "Cannot open 'postgresql-x64-18' service".

    WHY
    ---
    Measured 2026-08-06 during the wiki_chunks partition migration: drive I:
    (Seagate IronWolf 7200 RPM SATA) was 77% busy at only 33.7 MB/s with a queue
    depth of 0.6. A sequential HDD manages 150-200 MB/s, so that profile is SEEK
    contention, not bandwidth: source reads, partition writes, TOAST and WAL are
    all competing for one spindle. Moving WAL to a separate NVMe removes one
    whole class of writes from that contention.

    SAFETY
    ------
    Nothing is deleted. The original pg_wal is renamed to pg_wal_old and left in
    place; this script tells you to remove it manually only after the server is
    verified healthy. The WAL is COPIED (robocopy) before anything is renamed,
    so an interrupted run leaves the original intact.

    A clean shutdown is verified via pg_controldata before pg_wal is touched.
    Moving WAL after an UNCLEAN shutdown can destroy the segments crash recovery
    needs — on this host, which reboots under load, that check is not optional.

.EXAMPLE
    # In an elevated PowerShell:
    powershell -ExecutionPolicy Bypass -File F:\Projects\sage_kaizen_ai\scripts\move_pg_wal_to_nvme.ps1
#>
[CmdletBinding()]
param(
    [string]$Service  = 'postgresql-x64-18',
    [string]$PgData   = 'I:\Program Files\PostgreSQL\18\data',
    [string]$NewWal   = 'E:\pgwal',
    [string]$BinDir   = 'C:\Program Files\PostgreSQL\18\bin'
)

$ErrorActionPreference = 'Stop'

function Fail($msg) { Write-Host "FAILED: $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# --- 0. Preconditions ------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal] `
            [Security.Principal.WindowsIdentity]::GetCurrent()
           ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Fail "not elevated. Re-run this in an Administrator PowerShell." }

$oldWal = Join-Path $PgData 'pg_wal'
$bakWal = Join-Path $PgData 'pg_wal_old'

if (-not (Test-Path $PgData)) { Fail "PGDATA not found: $PgData" }
if (-not (Test-Path $oldWal)) { Fail "pg_wal not found: $oldWal" }
if (Test-Path $bakWal)        { Fail "$bakWal already exists — a previous run left state. Resolve by hand." }

# A junction already in place means this has been done.
if ((Get-Item $oldWal).LinkType) { Fail "$oldWal is already a link — nothing to do." }

$walBytes = (Get-ChildItem $oldWal -Recurse -File | Measure-Object Length -Sum).Sum
$freeE    = (Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$($NewWal.Substring(0,2))'").FreeSpace
Write-Host ("pg_wal size: {0:N1} GB   target free: {1:N1} GB" -f ($walBytes/1GB), ($freeE/1GB))
if ($freeE -lt ($walBytes * 3)) { Fail "not enough free space on $NewWal (want 3x WAL size for headroom)" }

# --- 1. Stop the service ---------------------------------------------------
Step "Stopping $Service"
Stop-Service $Service -Force
$deadline = (Get-Date).AddMinutes(5)
while ((Get-Service $Service).Status -ne 'Stopped' -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
}
if ((Get-Service $Service).Status -ne 'Stopped') { Fail "service did not stop within 5 minutes" }
Start-Sleep -Seconds 3
Write-Host "service stopped"

# --- 2. Verify a CLEAN shutdown -------------------------------------------
Step "Verifying clean shutdown (pg_controldata)"
$state = & (Join-Path $BinDir 'pg_controldata.exe') -D $PgData |
         Select-String 'Database cluster state:'
Write-Host "  $state"
if ($state -notmatch 'shut down') {
    Start-Service $Service
    Fail "cluster state is not 'shut down'. Service restarted, pg_wal untouched. Investigate before retrying."
}

# --- 3. Copy WAL to the new location --------------------------------------
Step "Copying WAL to $NewWal (nothing is deleted)"
if (-not (Test-Path $NewWal)) { New-Item -ItemType Directory -Path $NewWal | Out-Null }
# /MIR mirrors; /COPYALL preserves ACLs+timestamps; /NFL /NDL quiet the listing.
robocopy $oldWal $NewWal /MIR /COPYALL /R:2 /W:2 /NFL /NDL /NJH | Out-Null
# robocopy exit codes below 8 are success (0-7 = copied / extra / mismatched).
if ($LASTEXITCODE -ge 8) {
    Start-Service $Service
    Fail "robocopy failed with exit code $LASTEXITCODE. Service restarted, pg_wal untouched."
}
$srcCount = (Get-ChildItem $oldWal -File).Count
$dstCount = (Get-ChildItem $NewWal -File).Count
Write-Host "  segments: source=$srcCount  target=$dstCount"
if ($dstCount -lt $srcCount) {
    Start-Service $Service
    Fail "target has fewer files than source. Service restarted, pg_wal untouched."
}

# --- 4. Permissions for the service account -------------------------------
Step "Granting the service account access to $NewWal"
$acct = (Get-CimInstance Win32_Service -Filter "Name='$Service'").StartName
Write-Host "  service runs as: $acct"
& icacls $NewWal /grant "${acct}:(OI)(CI)F" /T | Select-Object -Last 1

# --- 5. Swap in the junction ----------------------------------------------
Step "Renaming original to pg_wal_old and creating the junction"
Rename-Item -Path $oldWal -NewName 'pg_wal_old'
& cmd /c mklink /J "`"$oldWal`"" "`"$NewWal`"" | Out-Null
$link = Get-Item $oldWal
if (-not $link.LinkType) {
    Rename-Item -Path $bakWal -NewName 'pg_wal'
    Start-Service $Service
    Fail "junction was not created. Original restored, service restarted."
}
Write-Host "  $oldWal -> $($link.Target)"

# --- 6. Start and verify ---------------------------------------------------
Step "Starting $Service"
Start-Service $Service
$deadline = (Get-Date).AddMinutes(10)
while ((Get-Service $Service).Status -ne 'Running' -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
}
if ((Get-Service $Service).Status -ne 'Running') {
    Write-Host "service did not start. To roll back:" -ForegroundColor Yellow
    Write-Host "  Remove-Item '$oldWal' -Force"
    Write-Host "  Rename-Item '$bakWal' -NewName 'pg_wal'"
    Write-Host "  Start-Service $Service"
    Fail "service failed to start after the move"
}

Write-Host "`nDone. PostgreSQL is running with pg_wal on $NewWal." -ForegroundColor Green
Write-Host @"

NEXT STEPS
  1. Confirm the server is healthy and WAL is being written to the new path.
     psql is NOT on PATH by default on this machine, so call it by full path:
       & '$BinDir\psql.exe' -U postgres -d sage_kaizen -c "SELECT pg_walfile_name(pg_current_wal_lsn());"
       Get-ChildItem '$NewWal' | Sort-Object LastWriteTime -Descending | Select-Object -First 3

     Or without psql at all — the newest file in '$NewWal' having a timestamp
     from the last few minutes is sufficient evidence that WAL moved:
       Get-ChildItem '$NewWal' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 3 Name,LastWriteTime

  2. Resume the migration (it restarts from the last committed batch):
       cd F:\Projects\sage_kaizen_ai
       python scripts/migrate_wiki_chunks_partitioned.py --copy --batch-rows 100000

  3. ONLY after the server has run healthily for a while, reclaim the old copy:
       Remove-Item '$bakWal' -Recurse -Force
"@
