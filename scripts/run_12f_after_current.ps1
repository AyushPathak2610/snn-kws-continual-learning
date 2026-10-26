# Waits until the running snn_gscv2_tradeoff.ipynb run has finished, then runs the notebook again on the real
# dataset. Finished stages and runs are loaded from runs/gscv2_tradeoff, so only what is missing is computed:
# the rest of stage 12e if the first run was interrupted, then the new stage 12f and the summary.
# Needs no internet. Progress: runs/gscv2_tradeoff/log.txt; status lines: runs/gscv2_tradeoff/scheduled_run_status.txt

$proj   = "D:\codes\Anmol papers and code\heban_fischer_mi"
$py     = "C:\Users\Ayush\anaconda3\envs\ai-risk-oracle\python.exe"
$status = Join-Path $proj "runs\gscv2_tradeoff\scheduled_run_status.txt"

function Note($msg) { Add-Content -Path $status -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" -Encoding utf8 }

Note "scheduled task started; waiting for the current tradeoff run to finish"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -like '*run_nb.py*snn_gscv2_tradeoff*' }) {
    Start-Sleep -Seconds 60
}
Note "no tradeoff run is active; starting the notebook on the real dataset (stage 12f)"
$p = Start-Process -FilePath $py `
        -ArgumentList '"scripts\run_nb.py"', 'snn_gscv2_tradeoff.ipynb', '--real' `
        -WorkingDirectory $proj -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $proj "runs\gscv2_tradeoff\scheduled_run_stdout.txt") `
        -RedirectStandardError  (Join-Path $proj "runs\gscv2_tradeoff\scheduled_run_stderr.txt")
Note "notebook finished with exit code $($p.ExitCode)"
