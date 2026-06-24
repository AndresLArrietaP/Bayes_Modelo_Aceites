# Setup del entorno en Windows (PowerShell)
# Ejecutar desde la raíz del proyecto:  .\scripts\setup.ps1

Write-Host "== Creando entorno virtual (venv) ==" -ForegroundColor Cyan
python -m venv venv
.\venv\Scripts\Activate.ps1

Write-Host "== Actualizando pip e instalando dependencias ==" -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Se creó .env desde la plantilla. EDÍTALO con tus credenciales." -ForegroundColor Yellow
}

Write-Host "`nListo. Recuerda instalar 'ODBC Driver 18 for SQL Server' para conectar a Azure." -ForegroundColor Green
Write-Host "Prueba rapida (datos sinteticos):  python -m src.train ; python -m src.predict"
