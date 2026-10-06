"""Contrato de SearchInput y de la herramienta search_knowledge_base."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import main as herramienta
from main import (
    SearchInput,
    db,
    ejecutar_tool_call,
    search_knowledge_base,
)


def test_args_schema_es_search_input() -> None:
    assert search_knowledge_base.args_schema is SearchInput
    assert search_knowledge_base.name == "search_knowledge_base"
    descripcion = search_knowledge_base.description
    assert "base de conocimientos técnica" in descripcion
    assert "No la uses" in descripcion
    assert "Descripción de la herramienta aquí" not in descripcion
    assert search_knowledge_base.handle_validation_error is not False


def test_search_input_rechaza_query_vacia() -> None:
    with pytest.raises(ValidationError, match="query no puede estar vacía"):
        SearchInput(query="   ", limit=3)


def test_search_input_rechaza_limit_cero() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        SearchInput(query="vpn", limit=0)


def test_search_input_exige_query() -> None:
    with pytest.raises(ValidationError):
        SearchInput(limit=3)


def test_search_input_rechaza_limit_no_entero() -> None:
    with pytest.raises(ValidationError):
        SearchInput(query="vpn", limit="muchos")


def test_search_input_no_coerciona_limit_numerico_en_texto() -> None:
    with pytest.raises(ValidationError, match="valid integer"):
        SearchInput(query="vpn", limit="3")


def test_search_input_rechaza_limit_sobre_el_tope() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 10"):
        SearchInput(query="vpn", limit=11)


def test_search_input_rechaza_campo_extra() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SearchInput(query="vpn", limit=3, sql="select 1")


async def test_limit_llega_como_k() -> None:
    texto = await search_knowledge_base.ainvoke(
        {"query": "  correo de la sede  ", "limit": 7}
    )
    assert db.llamadas == [("correo de la sede", 7)]
    assert "k=7" in texto
    assert "Resultado para correo de la sede" in texto


async def test_default_limit_es_3() -> None:
    await search_knowledge_base.ainvoke({"query": "vpn"})
    assert db.llamadas == [("vpn", 3)]


async def test_ciclo_tool_call_valida_y_ejecuta() -> None:
    tool_call = {
        "name": "search_knowledge_base",
        "args": {"query": "sede", "limit": 4},
    }
    texto = await ejecutar_tool_call(tool_call["name"], tool_call["args"])
    assert db.llamadas == [("sede", 4)]
    assert "k=4" in texto
    assert "Resultado para sede" in texto


async def test_herramienta_desconocida_no_toca_la_base() -> None:
    texto = await ejecutar_tool_call("drop_table", {"query": "x", "limit": 1})
    assert texto.startswith("Error controlado: herramienta desconocida:")
    assert db.llamadas == []


async def test_validacion_de_la_llamada_vuelve_como_string() -> None:
    texto = await ejecutar_tool_call(
        "search_knowledge_base",
        {"query": "vpn", "limit": 0},
    )
    assert texto.startswith("Error de validación: limit:")
    assert db.llamadas == []


async def test_ainvoke_invalido_no_toca_la_base() -> None:
    texto = await search_knowledge_base.ainvoke({"query": "vpn", "sql": "select 1"})
    assert "Extra inputs are not permitted" in texto
    assert db.llamadas == []


async def test_campo_extra_en_tool_call_no_toca_la_base() -> None:
    texto = await ejecutar_tool_call(
        "search_knowledge_base",
        {"query": "vpn", "limit": 3, "sql": "select 1"},
    )
    assert texto.startswith("Error de validación:")
    assert "Extra inputs are not permitted" in texto
    assert db.llamadas == []


async def test_coroutine_tambien_valida_con_search_input() -> None:
    texto = await search_knowledge_base.coroutine(query="   ", limit=3)
    assert texto.startswith("Error de validación:")
    assert "query no puede estar vacía" in texto
    assert db.llamadas == []


def test_esquema_declara_query_y_limit() -> None:
    esquema = SearchInput.model_json_schema()
    propiedades = esquema["properties"]
    assert set(propiedades) == {"query", "limit"}
    assert propiedades["limit"]["default"] == 3
    assert propiedades["limit"]["minimum"] == 1
    assert propiedades["limit"]["maximum"] == 10
    assert propiedades["query"]["minLength"] == 1
    assert propiedades["query"]["maxLength"] == 2000
    assert esquema["additionalProperties"] is False
    assert "procedimiento" in propiedades["query"]["description"]
