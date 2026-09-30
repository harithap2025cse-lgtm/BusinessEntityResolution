@echo off

echo Running Business Entity Resolution Challenge - 10K pipeline...
echo.

python -m src.main --limit 10000

if errorlevel 1 (
    echo.
    echo Pipeline failed.
    exit /b 1
)

echo.
echo Pipeline completed successfully.