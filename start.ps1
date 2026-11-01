Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting KernelForge via Docker Compose (One-Click)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Ensure .env exists
if (-not (Test-Path ".env")) {
    Write-Host "[.env not found - creating from .env.example]" -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

Write-Host "Building and launching containers..." -ForegroundColor Green
docker compose up --build -d

Write-Host "`nKernelForge services are healthy and running:" -ForegroundColor Green
Write-Host "  • Frontend Canvas:  http://localhost:3000" -ForegroundColor White
Write-Host "  • FastAPI Gateway:  http://localhost:8000" -ForegroundColor White
Write-Host "  • Swagger API Docs: http://localhost:8000/docs`n" -ForegroundColor White

Start-Process "http://localhost:3000"

