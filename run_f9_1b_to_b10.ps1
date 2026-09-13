# F9-1B full automation: resume from checkpoint (B5) through B10.
# Frozen protocol: docs/F9_1B_PREREGISTERED_ALLOCATOR_PROTOCOL_01.md
# Hash: 323fda6381c065db0c7dc72d2d665b84ae7828cb58beca52d49e6c7574580411
# Expected wall time: ~6.4 h for remaining 12 runs (~32 min/run).
$ErrorActionPreference = "Stop"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = "logs/f9_1b_full_run_$Stamp.log"

Start-Transcript -Path $LogFile -Append
try {
    Write-Host "=== F9-1B B5-B10 starting $Stamp ==="
    $Py = ".\.venv\Scripts\python.exe"
    & $Py f9_1b_allocator_reduction_runner.py `
        --checkpoint logs/f9_1b_checkpoint.json `
        --output logs/f9_1b_telemetry.json `
        --report docs/F9_1B_ALLOCATOR_REDUCTION_REPORT_01.md
    if ($LASTEXITCODE -ne 0) { throw "Runner exited with code $LASTEXITCODE. Checkpoint preserved; relaunch this same script to resume." }
    Write-Host "=== Campaign done. Regenerating analysis (no GPU) ==="
    & $Py f9_1b_allocator_reduction_runner.py --analyze-only `
        --checkpoint logs/f9_1b_checkpoint.json `
        --output logs/f9_1b_telemetry.json `
        --report docs/F9_1B_ALLOCATOR_REDUCTION_REPORT_01.md
    Write-Host "=== DONE ==="
}
finally {
    Stop-Transcript
}
