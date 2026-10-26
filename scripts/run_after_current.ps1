# Waits until no snn_gscv2_tradeoff.ipynb run is active, then runs the notebook on the real dataset. Finished
# stages and runs are loaded from runs/gscv2_tradeoff, so only what is missing is computed. Needs no internet.
# Usage: powershell -File run_after_current.ps1 -Tag 12g
# Progress: runs/gscv2_tradeoff/log.txt; status: runs/gscv2_tradeoff/scheduled_<Tag>_status.txt
param([string]$Tag = "next")

$proj   = "D:\codes\Anmol papers and code\heban_fischer_mi"
$py     = "C:\Users\Ayush\anaconda3\envs\ai-risk-oracle\python.exe"
$runDir = Join-Path $proj "runs\gscv2_tradeoff"
$status = Join-Path $runDir "scheduled_${Tag}_status.txt"

function Note($msg) { Add-Content -Path $status -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" -Encoding utf8 }

Note "scheduled task started; waiting for any running tradeoff notebook to finish"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -like '*run_nb.py*snn_gscv2_tradeoff*' }) {
    Start-Sleep -Seconds 60
}
Note "no tradeoff run is active; starting the notebook on the real dataset ($Tag)"
$p = Start-Process -FilePath $py `
        -ArgumentList '"scripts\run_nb.py"', 'snn_gscv2_tradeoff.ipynb', '--real' `
        -WorkingDirectory $proj -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $runDir "scheduled_${Tag}_stdout.txt") `
        -RedirectStandardError  (Join-Path $runDir "scheduled_${Tag}_stderr.txt")
Note "notebook finished with exit code $($p.ExitCode)"
