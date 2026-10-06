"""Familias de error: 401/key, 429/cuota, red/timeout, salida truncada."""

from __future__ import annotations

import pytest

import main as herramienta
from main import (
    ErrorBusqueda,
    ErrorClave,
    ErrorCuota,
    ErrorRed,
    ErrorSalidaTruncada,
    FalloSimulado,
    MAX_FRAGMENTO,
    clasificar_error_busqueda,
    db,
    search_knowledge_base,
)


def test_clasificar_401_lanza_error_clave() -> None:
    exc = clasificar_error_busqueda(
        FalloSimulado(401, "Unauthorized: invalid API key")
    )
    assert isinstance(exc, ErrorClave)
    assert str(exc).startswith("401/key: clave rechazada o ausente.")


def test_clasificar_429_lanza_error_cuota() -> None:
    exc = clasificar_error_busqueda(
        FalloSimulado(429, "rate limit / quota exceeded")
    )
    assert isinstance(exc, ErrorCuota)
    assert str(exc).startswith("429/cuota:")


def test_clasificar_timeout_lanza_error_red() -> None:
    exc = clasificar_error_busqueda(TimeoutError("Request timed out"))
    assert isinstance(exc, ErrorRed)
    assert str(exc).startswith("red/timeout:")


def test_clasificar_truncado_no_es_transitorio() -> None:
    exc = clasificar_error_busqueda(
        RuntimeError("finish_reason=length")
    )
    assert isinstance(exc, ErrorSalidaTruncada)
    assert str(exc).startswith("Salida truncada:")


async def test_401_devuelve_string_y_no_reintenta() -> None:
    db.fallos.append(FalloSimulado(401, "Unauthorized: invalid API key"))
    texto = await search_knowledge_base.ainvoke({"query": "vpn", "limit": 3})
    assert texto == (
        "401/key: clave rechazada o ausente. Detalle: Unauthorized: invalid API key "
        "No reintentes: la clave fue rechazada."
    )
    assert len(db.llamadas) == 1


async def test_429_reintenta_y_luego_falla(monkeypatch: pytest.MonkeyPatch) -> None:
    esperas: list[float] = []

    async def _anotar(segundos: float) -> None:
        esperas.append(segundos)

    monkeypatch.setattr(herramienta.asyncio, "sleep", _anotar)
    db.fallos.extend(
        [
            FalloSimulado(429, "rate limit / quota exceeded"),
            FalloSimulado(429, "rate limit / quota exceeded"),
            FalloSimulado(429, "rate limit / quota exceeded"),
        ]
    )
    texto = await search_knowledge_base.ainvoke({"query": "backup", "limit": 3})
    assert texto == (
        "429/cuota: la API rechazó la búsqueda por cuota o rate limit. "
        "Detalle: rate limit / quota exceeded "
        "Reintentá la consulta más tarde."
    )
    assert len(db.llamadas) == 3
    assert esperas == [0.5, 1.0]


async def test_429_recupera_en_el_segundo_intento() -> None:
    db.fallos.append(FalloSimulado(429, "rate limit / quota exceeded"))
    texto = await search_knowledge_base.ainvoke({"query": "backup", "limit": 1})
    assert "Resultado para backup" in texto
    assert "k=1" in texto
    assert len(db.llamadas) == 2


async def test_timeout_agotado() -> None:
    db.fallos.extend(
        [
            TimeoutError("Request timed out talking to the index"),
            TimeoutError("Request timed out talking to the index"),
            TimeoutError("Request timed out talking to the index"),
        ]
    )
    texto = await search_knowledge_base.ainvoke({"query": "vpn", "limit": 2})
    assert texto == (
        "red/timeout: no se pudo hablar con la base vectorial. "
        "Detalle: Request timed out talking to the index "
        "Reintentá la consulta."
    )
    assert len(db.llamadas) == 3


async def test_timeout_recupera_en_el_segundo_intento() -> None:
    db.fallos.append(ConnectionError("connection reset"))
    texto = await search_knowledge_base.ainvoke({"query": "dns", "limit": 2})
    assert "Resultado para dns" in texto
    assert len(db.llamadas) == 2


async def test_salida_truncada_no_recorta_ni_reintenta() -> None:
    db.respuesta = ["x" * (MAX_FRAGMENTO + 1)]
    texto = await search_knowledge_base.ainvoke({"query": "log", "limit": 3})
    assert texto == (
        f"Salida truncada: el fragmento 0 supera {MAX_FRAGMENTO} "
        "caracteres y no se recorta en silencio. "
        "Reducí el limit y reformulá la consulta."
    )
    assert len(db.llamadas) == 1


async def test_mas_fragmentos_que_k_no_se_recorta() -> None:
    db.respuesta = ["uno", "dos", "tres"]
    texto = await search_knowledge_base.ainvoke({"query": "log", "limit": 2})
    assert texto.startswith("Salida truncada:")
    assert "limit/k pedido fue 2" in texto
    assert len(db.llamadas) == 1


async def test_salida_no_es_lista() -> None:
    db.respuesta = "cortado"  # type: ignore[assignment]
    texto = await search_knowledge_base.ainvoke({"query": "log", "limit": 3})
    assert texto.startswith("Salida truncada: similarity_search no devolvió")
    assert len(db.llamadas) == 1


async def test_error_no_transitorio_no_reintenta() -> None:
    db.fallos.append(RuntimeError("boom"))
    texto = await search_knowledge_base.ainvoke({"query": "vpn", "limit": 3})
    assert texto == "Error de búsqueda: boom Reformulá la consulta."
    assert len(db.llamadas) == 1
    assert issubclass(ErrorBusqueda, Exception)


def test_clases_de_error_no_se_confunden() -> None:
    assert issubclass(ErrorClave, ErrorBusqueda)
    assert issubclass(ErrorCuota, ErrorBusqueda)
    assert issubclass(ErrorRed, ErrorBusqueda)
    assert issubclass(ErrorSalidaTruncada, ErrorBusqueda)
    assert not issubclass(ErrorClave, ErrorCuota)
