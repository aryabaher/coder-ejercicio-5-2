"""Deja el mock limpio y sin esperas reales entre tests."""

from __future__ import annotations

import pytest

import main as herramienta


@pytest.fixture(autouse=True)
def _db_limpia(monkeypatch: pytest.MonkeyPatch) -> None:
    herramienta.db.fallos.clear()
    herramienta.db.llamadas.clear()
    herramienta.db.respuesta = None

    async def _sin_espera(_segundos: float) -> None:
        return None

    monkeypatch.setattr(herramienta.asyncio, "sleep", _sin_espera)
    yield
    herramienta.db.fallos.clear()
    herramienta.db.llamadas.clear()
    herramienta.db.respuesta = None
