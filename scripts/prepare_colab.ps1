# Script tao file zip de upload len Google Colab
# Chay tu thu muc goc: c:\vqa project

$ErrorActionPreference = "Stop"
$zipPath = "vqa_project.zip"

if (Test-Path $zipPath) {
    Remove-Item $zipPath
    Write-Host "Da xoa zip cu"
}

$items = @("app", "src", "data", "artifacts", "requirements.txt", "pyproject.toml")

Write-Host "Dang dong goi project..."
Compress-Archive -Path $items -DestinationPath $zipPath -CompressionLevel Optimal

$sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host "Xong! File: $zipPath ($sizeMB MB)"
Write-Host ""
Write-Host "Buoc tiep theo:"
Write-Host "  1. Mo Google Colab: https://colab.research.google.com"
Write-Host "  2. Upload file vqa_train_colab.ipynb"
Write-Host "  3. Chay notebook va upload vqa_project.zip khi duoc hoi"
