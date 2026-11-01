@echo off
echo ========================================================
echo  Starting KernelForge (Backend + Frontend) via Docker
echo ========================================================

REM Ensure .env exists
if not exist .env (
    echo [.env not found - copying from .env.example]
    copy .env.example .env
)

echo.
echo Building and launching Docker containers...
docker compose up --build -d

echo.
echo ========================================================
echo  KernelForge is running!
echo   - Frontend: http://localhost:3000
echo   - Backend API: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo ========================================================
echo Opening browser...
start http://localhost:3000

echo.
echo To view live logs: docker compose logs -f
echo To stop: docker compose down
pause

