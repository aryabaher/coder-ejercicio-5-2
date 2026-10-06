"""Tool calling: search_knowledge_base sobre una base vectorial simulada.

Contrato de la herramienta
- Nombre: search_knowledge_base
- Descripción: busca fragmentos en la base de conocimientos técnica de soporte.
- Esquema de Entrada: SearchInput (query: str, limit: int entre 1 y 10, default 3). Sin campos extra.
- Esquema de Salida: str con los fragmentos, o un error controlado para el rol tool.

El modelo no ejecuta código: emite tool_calls. Esta aplicación valida con
Pydantic y recién ahí llama a similarity_search. La herramienta solo lee
(query y limit). No acepta SQL ni un filtro armado por el modelo.

Reintentos: 429/cuota y red/timeout, hasta 3 intentos (0.5s, 1s, 2s).
401/key y salida truncada no se reintentan.
"""

from __future__ import annotations

import asyncio
import sys
from typing import List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MAX_INTENTOS = 3
ESPERA_INICIAL_S = 0.5
MAX_FRAGMENTO = 20000
MAX_QUERY = 2000
MAX_K = 10


class ErrorBusqueda(Exception):
    """Fallo controlado de la búsqueda. Se devuelve como string al modelo."""


class ErrorClave(ErrorBusqueda):
    """401/key: credencial rechazada. No se reintenta."""


class ErrorCuota(ErrorBusqueda):
    """429/cuota: rate limit o cupo. Se reintenta."""


class ErrorRed(ErrorBusqueda):
    """red/timeout: no hubo respuesta de la base. Se reintenta."""


class ErrorSalidaTruncada(ErrorBusqueda):
    """El fragmento vino cortado o supera el tope. No se recorta ni se reintenta."""


class FalloSimulado(Exception):
    """Error del mock (status HTTP). No sale a ninguna API real."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(detail)


class SearchInput(BaseModel):
    """Esquema de entrada de search_knowledge_base. Tipos estrictos, sin campos extra."""

    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY,
        description=(
            "Texto a buscar en la base de conocimientos técnica: un procedimiento, "
            "un error conocido o un dato de configuración interna. No puede estar vacío."
        ),
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=MAX_K,
        description=(
            f"Cantidad máxima de fragmentos a devolver. Entero entre 1 y {MAX_K}. "
            "Si se omite, se devuelven 3."
        ),
    )

    @field_validator("query")
    @classmethod
    def query_no_vacia(cls, valor: str) -> str:
        limpio = valor.strip()
        if not limpio:
            raise ValueError("query no puede estar vacía.")
        return limpio


class MockVectorDB:
    """Simulación de base de datos vectorial. No llama a ninguna API real."""

    def __init__(self) -> None:
        self.fallos: List[BaseException] = []
        self.llamadas: List[tuple] = []
        self.respuesta: Optional[List[str]] = None

    async def similarity_search(self, query: str, k: int) -> List[str]:
        self.llamadas.append((query, k))
        if self.fallos:
            raise self.fallos.pop(0)
        if self.respuesta is not None:
            salida = self.respuesta
            self.respuesta = None
            return salida
        return [f"Resultado para {query}"]


db = MockVectorDB()


def _status(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def clasificar_error_busqueda(exc: BaseException) -> ErrorBusqueda:
    """Traduce el fallo de la base a un error de la herramienta."""

    if isinstance(exc, ErrorBusqueda):
        return exc
    detalle = str(exc).strip() or exc.__class__.__name__
    texto = detalle.lower()
    status = _status(exc)
    if "finish_reason=length" in texto or "salida truncada" in texto:
        return ErrorSalidaTruncada(
            f"Salida truncada: la base cortó la respuesta. Detalle: {detalle}"
        )
    if status == 401 or "invalid api key" in texto or "unauthorized" in texto:
        return ErrorClave(f"401/key: clave rechazada o ausente. Detalle: {detalle}")
    if status == 429 or "rate limit" in texto or "quota" in texto:
        return ErrorCuota(
            f"429/cuota: la API rechazó la búsqueda por cuota o rate limit. Detalle: {detalle}"
        )
    if (
        isinstance(exc, (TimeoutError, ConnectionError))
        or "timed out" in texto
        or "timeout" in texto
    ):
        return ErrorRed(
            f"red/timeout: no se pudo hablar con la base vectorial. Detalle: {detalle}"
        )
    return ErrorBusqueda(f"Error de búsqueda: {detalle}")


def validar_salida(resultados: object, k: int) -> List[str]:
    """Rechaza una salida cortada. k es el limit pedido: no se recorta la lista."""

    if k < 1:
        raise ErrorBusqueda(f"limit/k inválido: {k}.")
    if not isinstance(resultados, list):
        raise ErrorSalidaTruncada(
            "Salida truncada: similarity_search no devolvió la lista completa de fragmentos."
        )
    if len(resultados) > k:
        raise ErrorSalidaTruncada(
            f"Salida truncada: la base devolvió {len(resultados)} fragmentos "
            f"y el limit/k pedido fue {k}. No se recorta en silencio."
        )
    textos: List[str] = []
    for indice, item in enumerate(resultados):
        if not isinstance(item, str):
            raise ErrorSalidaTruncada(
                f"Salida truncada: el fragmento {indice} no es texto completo."
            )
        if len(item) > MAX_FRAGMENTO:
            raise ErrorSalidaTruncada(
                f"Salida truncada: el fragmento {indice} supera {MAX_FRAGMENTO} "
                "caracteres y no se recorta en silencio."
            )
        textos.append(item)
    return textos


def formatear_resultados(fragmentos: List[str], limit: int) -> str:
    """Esquema de salida: texto que el modelo lee como observación (rol tool)."""

    lineas = [
        f"[{indice}] {texto}" for indice, texto in enumerate(fragmentos, start=1)
    ]
    return f"Fragmentos (k={limit}):\n" + "\n".join(lineas)


def _mensaje_validacion(exc: ValidationError) -> str:
    err = exc.errors()[0]
    lugar = ".".join(str(parte) for parte in err["loc"]) or "args"
    return f"Error de validación: {lugar}: {err['msg']}"


async def con_reintentos(query: str, limit: int) -> List[str]:
    """Llama a similarity_search con k=limit. Reintenta solo 429 y red/timeout."""

    espera = ESPERA_INICIAL_S
    ultimo: Optional[ErrorBusqueda] = None
    for intento in range(1, MAX_INTENTOS + 1):
        try:
            resultados = await db.similarity_search(query, k=limit)
            return validar_salida(resultados, k=limit)
        except Exception as exc:
            clasificado = clasificar_error_busqueda(exc)
            transitorio = isinstance(clasificado, (ErrorCuota, ErrorRed))
            if not transitorio or intento == MAX_INTENTOS:
                raise clasificado from exc
            ultimo = clasificado
            await asyncio.sleep(espera)
            espera *= 2
    if ultimo is None:
        raise ErrorBusqueda("Error de búsqueda: reintentos agotados sin detalle.")
    raise ultimo


@tool(args_schema=SearchInput)
async def search_knowledge_base(query: str, limit: int = 3) -> str:
    """Busca procedimientos, errores conocidos y configuración en la base de conocimientos técnica de soporte.

    Usala solo si la pregunta pide cómo hacer una tarea documentada, qué significa
    un error de esa plataforma o cómo está configurado un sistema interno.
    No la uses para cálculos, precios, clima, charla general, ni para ejecutar SQL
    o consultar un sistema que no esté en esa base.
    """

    try:
        datos = SearchInput(query=query, limit=limit)
        fragmentos = await con_reintentos(datos.query, datos.limit)
    except ValidationError as exc:
        return _mensaje_validacion(exc)
    except ErrorBusqueda as exc:
        return str(exc)
    return formatear_resultados(fragmentos, datos.limit)


search_knowledge_base.handle_validation_error = _mensaje_validacion


async def ejecutar_tool_call(nombre: str, args: dict) -> str:
    """Paso local del ciclo: valida con Pydantic, ejecuta y devuelve la observación."""

    if nombre != search_knowledge_base.name:
        return f"Error controlado: herramienta desconocida: {nombre}"
    try:
        SearchInput.model_validate(args)
    except ValidationError as exc:
        return _mensaje_validacion(exc)
    return await search_knowledge_base.ainvoke(args)


def _exigir(condicion: bool, mensaje: str) -> None:
    if not condicion:
        print(f"Error controlado: {mensaje}")
        raise SystemExit(1)


async def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    db.fallos.clear()
    db.llamadas.clear()
    db.respuesta = None

    print(f"nombre={search_knowledge_base.name}")
    print(f"descripcion={search_knowledge_base.description.strip().splitlines()[0]}")
    print(f"args_schema={search_knowledge_base.args_schema.__name__}")
    print(f"entrada={list(SearchInput.model_json_schema()['properties'])}")

    tool_call = {
        "name": "search_knowledge_base",
        "args": {"query": "caída del correo de la sede", "limit": 2},
    }
    observacion = await ejecutar_tool_call(tool_call["name"], tool_call["args"])
    print(f"tool_calls: [{tool_call}]")
    print(f"rol tool: {observacion}")
    _exigir(
        db.llamadas == [("caída del correo de la sede", 2)],
        f"limit no llegó como k: {db.llamadas}",
    )
    _exigir("Resultado para caída del correo de la sede" in observacion, observacion)
    _exigir("k=2" in observacion, observacion)

    db.llamadas.clear()
    por_defecto = await search_knowledge_base.ainvoke({"query": "vpn"})
    print(f"default_k={db.llamadas[-1][1]}")
    print(por_defecto)
    _exigir(db.llamadas == [("vpn", 3)], f"el default 3 no llegó como k: {db.llamadas}")

    invalida = await ejecutar_tool_call(
        "search_knowledge_base",
        {"query": "vpn", "limit": 0},
    )
    print(invalida)
    _exigir(invalida.startswith("Error de validación:"), invalida)

    db.fallos.append(FalloSimulado(401, "Unauthorized: invalid API key"))
    db.llamadas.clear()
    error_clave = await search_knowledge_base.ainvoke({"query": "vpn", "limit": 3})
    print(error_clave)
    print(f"intentos_401={len(db.llamadas)}")
    _exigir(error_clave.startswith("401/key:"), error_clave)
    _exigir(len(db.llamadas) == 1, "401/key no debe reintentarse")

    db.fallos.append(FalloSimulado(429, "rate limit / quota exceeded"))
    db.llamadas.clear()
    recuperada = await search_knowledge_base.ainvoke({"query": "backup", "limit": 1})
    print(recuperada)
    print(f"intentos_429={len(db.llamadas)}")
    _exigir("Resultado para backup" in recuperada, recuperada)
    _exigir(len(db.llamadas) == 2, "429/cuota debía reintentar y recuperar")

    db.respuesta = ["x" * (MAX_FRAGMENTO + 1)]
    db.llamadas.clear()
    truncada = await search_knowledge_base.ainvoke({"query": "log enorme", "limit": 3})
    print(truncada)
    print(f"intentos_truncado={len(db.llamadas)}")
    _exigir(truncada.startswith("Salida truncada:"), truncada)
    _exigir(len(db.llamadas) == 1, "la salida truncada no se reintenta")


if __name__ == "__main__":
    asyncio.run(main())
