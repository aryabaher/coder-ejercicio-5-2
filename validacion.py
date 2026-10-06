"""Chequeo offline de la herramienta y la demo (sin API real)."""

from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path

from main import main

ROOT = Path(__file__).resolve().parent
FUENTE = (ROOT / "main.py").read_text(encoding="utf-8")

CADENAS = [
    "class SearchInput",
    "class MockVectorDB",
    "async def similarity_search(self, query: str, k: int)",
    'return [f"Resultado para {query}"]',
    "args_schema=SearchInput",
    "async def search_knowledge_base(query: str, limit: int = 3)",
    "await db.similarity_search(query, k=limit)",
    "Nombre: search_knowledge_base",
    "Esquema de Entrada",
    "Esquema de Salida",
    "401/key",
    "429/cuota",
    "red/timeout",
    "Salida truncada",
    "Reintentá la consulta más tarde.",
    "tool_calls",
    'if __name__ == "__main__":',
    "asyncio.run(main())",
]


def _configure_stdio() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


def _ok(nombre: str, condicion: bool, detalle: str = "") -> None:
    estado = "OK" if condicion else "FALLO"
    extra = f" - {detalle}" if detalle else ""
    print(f"[{estado}] {nombre}{extra}")
    if not condicion:
        raise SystemExit(1)


def _chequear_cuerpo(arbol: ast.Module) -> None:
    for nodo in arbol.body:
        if not isinstance(nodo, ast.AsyncFunctionDef):
            continue
        if nodo.name != "search_knowledge_base":
            continue
        inicio = nodo.decorator_list[0].lineno if nodo.decorator_list else nodo.lineno
        fin = nodo.end_lineno or nodo.lineno
        cuerpo = "\n".join(FUENTE.splitlines()[inicio - 1 : fin])
        _ok("la herramienta está al inicio del archivo", inicio < 80)
        for cadena in (
            "@tool",
            "args_schema=SearchInput",
            "await db.similarity_search(query, k=limit)",
            "429/cuota:",
            "Salida truncada:",
            "try:",
        ):
            _ok(f"dentro de search_knowledge_base: {cadena}", cadena in cuerpo)
        return
    _ok("existe search_knowledge_base", False)


def _chequear_cadenas() -> None:
    _ok("sin TODO", "TODO" not in FUENTE)
    arbol = ast.parse(FUENTE)
    _ok("main.py parsea", isinstance(arbol, ast.Module))
    for cadena in CADENAS:
        _ok(f"cadena {cadena}", cadena in FUENTE)
    _ok("k no está clavado en 3", "k=3" not in FUENTE)
    _ok("k no está clavado en 4", "k=4" not in FUENTE)
    _chequear_cuerpo(arbol)


def _correr_demo() -> None:
    asyncio.run(main())
    _ok("demo", True)


if __name__ == "__main__":
    _configure_stdio()
    _chequear_cadenas()
    _correr_demo()
