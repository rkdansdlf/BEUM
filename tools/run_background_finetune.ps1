# Run BEUM YOLOv8 segmentation fine-tuning in background before heading out to capture
param(
    [int]$Epochs = 30,
    [int]$Batch = 4
)

$RootDir = "C:\Project\BEUM"
$LogDir = Join-Path $RootDir "runs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$StdOutLog = Join-Path $LogDir "train_finetune.log"
$StdErrLog = Join-Path $LogDir "train_error.log"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Launching BEUM Background Fine-Tuning" -ForegroundColor Cyan
Write-Host " Epochs: $Epochs | Batch: $Batch" -ForegroundColor Cyan
Write-Host " Stdout log: $StdOutLog" -ForegroundColor Yellow
Write-Host " Stderr log: $StdErrLog" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

$process = Start-Process -FilePath "py" `
    -ArgumentList "-3.12 $RootDir\tools\train_yolo_seg_3class.py --epochs $Epochs --batch $Batch" `
    -WorkingDirectory $RootDir `
    -RedirectStandardOutput $StdOutLog `
    -RedirectStandardError $StdErrLog `
    -PassThru

Write-Host "`nProcess started! PID: $($process.Id)" -ForegroundColor Green
Write-Host "You can now safely head out for your camera capture session." -ForegroundColor Green
Write-Host "To monitor training progress, run:" -ForegroundColor White
Write-Host "  Get-Content runs\train_finetune.log -Wait -Tail 20`n" -ForegroundColor Gray
