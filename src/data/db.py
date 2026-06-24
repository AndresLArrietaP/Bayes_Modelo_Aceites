"""Conexión a Azure SQL vía SQLAlchemy.

La cadena de conexión se lee SIEMPRE desde la variable de entorno
DATABASE_URL (archivo .env, que está en .gitignore). Nunca se escribe
una credencial en el código fuente.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()  # carga .env si existe


def get_engine() -> Engine:
    """Crea un Engine de SQLAlchemy hacia Azure SQL.

    Requiere:
      - DATABASE_URL definido en el entorno (.env).
      - "ODBC Driver 18 for SQL Server" instalado en el sistema operativo.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL no está definido. Copia .env.example a .env y "
            "rellena tu cadena de conexión."
        )
    return create_engine(url, pool_pre_ping=True)
