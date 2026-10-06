# Búsqueda en la base de conocimientos

`search_knowledge_base` es una herramienta asíncrona de LangChain. El modelo no ejecuta código: arma un `tool_calls` con `query` y `limit`. Esta aplicación valida esos argumentos con `SearchInput` y recién ahí llama a `similarity_search`. El `limit` de la llamada llega como `k` (el default es 3; un `limit=7` no se reemplaza por 3).

La base es `MockVectorDB`: no hay API real.

## Quick path

1. Entorno e instalación. Tests: `pytest`, `pytest-asyncio`, `pytest-mock`. Un `pip install -r requirements.txt` alcanza.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux/macOS (bash/zsh):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Chequeo offline (mocks) y tests:

```
python validacion.py
python -m pytest -v
```

3. Demo (no pide clave: la base es el mock):

**Windows (PowerShell):**

```powershell
python main.py
```

**Linux/macOS (bash/zsh):**

```bash
python main.py
```

Deberías ver `nombre=search_knowledge_base`, `args_schema=SearchInput`, `tool_calls`, `k=2`, `default_k=3`, el `401/key` en un solo intento, la recuperación tras un `429/cuota` y la `Salida truncada` sin reintento. Si el `limit` no llega como `k`, imprime `Error controlado: ...` y sale con código 1.

## Archivos del repositorio

| Artefacto | Dónde está |
|-----------|------------|
| `SearchInput` (`query`, `limit`) | `main.py` |
| `MockVectorDB.similarity_search` | `main.py` |
| `@tool(args_schema=SearchInput)` + `async def search_knowledge_base` | `main.py` |
| `con_reintentos` (`k=limit`, backoff 429/red) | `main.py` |
| `ejecutar_tool_call` (valida y ejecuta el `tool_calls`) | `main.py` |
| Contrato nombre / descripción / esquemas | docstring de `main.py` |
| Demo ejecutable | `main.py` → `if __name__ == "__main__"` |
| Chequeo offline | `validacion.py` |
| Tests | `tests/` |
| `pytest.ini` | `testpaths = tests` · `asyncio_mode = auto` · `addopts = -ra -q` |
| `requirements.txt` | `langchain-core`, `pydantic`, `pytest`, `pytest-asyncio`, `pytest-mock` |

## Comportamiento

| Requisito | Cómo se cumple | Evidencia |
|-----------|----------------|-----------|
| `SearchInput` | `query` str (1 a 2000) y `limit` int (1 a 10, default 3). Tipos estrictos y sin campos extra. | `test_esquema_declara_query_y_limit` · `test_search_input_exige_query` · `test_search_input_rechaza_campo_extra` |
| `@tool` + esquema | `@tool(args_schema=SearchInput)` sobre la corrutina. | `test_args_schema_es_search_input` |
| Async | `async def search_knowledge_base`. Se invoca con `ainvoke`. | `test_limit_llega_como_k` |
| Docstring | Dice cuándo usarla (procedimiento, error o configuración de esa base) y cuándo no (cálculos, precios, SQL, charla). | `test_args_schema_es_search_input` |
| `k=limit` | `await db.similarity_search(query, k=limit)`. No hay `k=3` ni `k=4` clavados. | `test_limit_llega_como_k` · `test_default_limit_es_3` |
| Error como string | 401, 429, red y truncado vuelven en el texto de la observación. | `tests/test_errores.py` |
| Demo | `asyncio.run(main())` recorre el ciclo `tool_calls` → validación → observación. | `test_ciclo_tool_call_valida_y_ejecuta` · `evidencias/03-demo.txt` |

## Contrato de la herramienta

Queda en el docstring de `main.py`:

- **Nombre:** `search_knowledge_base`.
- **Descripción:** busca procedimientos, errores conocidos y configuración en la base de conocimientos técnica de soporte. Sirve cuando la pregunta pide cómo hacer una tarea documentada, qué significa un error de esa plataforma o cómo está configurado un sistema interno. No sirve para cálculos, precios, clima, charla general ni para ejecutar SQL.
- **Esquema de Entrada:** `SearchInput`, estricto y sin campos extra. `query` es texto de 1 a 2000 caracteres. `limit` es un entero de 1 a 10; si no viene, vale 3 y ese valor es el `k`.
- **Esquema de Salida:** un `str`. En éxito, `Fragmentos (k=<limit>)` y las líneas `[n] Resultado para ...`. Si falla, el mismo string trae el error para el rol `tool`.

## Manejo de errores personalizados

`search_knowledge_base` captura el fallo y lo devuelve como string. No hay traceback hacia el modelo. `ejecutar_tool_call` hace lo mismo con un argumento que no pasa `SearchInput`, y no llama a la base.

### Validación Pydantic

**Mensaje:** `Error de validación: limit: Input should be greater than or equal to 1`

Con query en blanco, `SearchInput` responde `Value error, query no puede estar vacía.` Si falta `query`: `Field required`. Si `limit` viene como texto, aunque sea `"3"`: `Input should be a valid integer`. Si `limit` es 11: `Input should be less than or equal to 10`. Si `query` supera 2000 caracteres: `String should have at most 2000 characters`. Un campo que no está en el esquema, por ejemplo `sql`: `Error de validación: sql: Extra inputs are not permitted`.

**Cómo reproducirlo:** `ejecutar_tool_call("search_knowledge_base", {"query": "vpn", "limit": 0})`. La base no se consulta (`llamadas` queda vacía). Lo mismo pasa con `ainvoke({"query": "vpn", "sql": "select 1"})`: el error vuelve como string y no hay llamada a `similarity_search`.

**Test:** `tests/test_herramienta.py::test_validacion_de_la_llamada_vuelve_como_string` · `test_ainvoke_invalido_no_toca_la_base` · `test_campo_extra_en_tool_call_no_toca_la_base` · `test_search_input_rechaza_query_vacia` · `test_search_input_rechaza_limit_cero` · `test_search_input_exige_query` · `test_search_input_rechaza_limit_no_entero` · `test_search_input_no_coerciona_limit_numerico_en_texto` · `test_search_input_rechaza_limit_sobre_el_tope` · `test_search_input_rechaza_campo_extra` · `test_coroutine_tambien_valida_con_search_input`

### 401 / key

**Mensaje:** `401/key: clave rechazada o ausente. Detalle: Unauthorized: invalid API key`

**Cómo reproducirlo:** encolar en `db.fallos` un `FalloSimulado(401, "Unauthorized: invalid API key")` y llamar `ainvoke`. No se reintenta: hay una sola llamada.

**Test:** `tests/test_errores.py::test_401_devuelve_string_y_no_reintenta` · `test_clasificar_401_lanza_error_clave`

### 429 / cuota

**Mensaje:** `429/cuota: la API rechazó la búsqueda por cuota o rate limit. Detalle: rate limit / quota exceeded`

**Cómo reproducirlo:** encolar tres `FalloSimulado(429, "rate limit / quota exceeded")`. Hay 3 llamadas; entre la primera y la segunda se espera 0.5s y entre la segunda y la tercera 1s (en tests la espera es 0, y un test anota `[0.5, 1.0]`). Si el segundo intento responde bien, vuelve `Resultado para ...`.

**Test:** `tests/test_errores.py::test_429_reintenta_y_luego_falla` · `test_429_recupera_en_el_segundo_intento`

### red / timeout

**Mensaje:** `red/timeout: no se pudo hablar con la base vectorial. Detalle: Request timed out talking to the index`

**Cómo reproducirlo:** encolar tres `TimeoutError("Request timed out talking to the index")`. Se reintenta igual que el 429. Un `ConnectionError` en el primer intento y éxito en el segundo devuelve el fragmento.

**Test:** `tests/test_errores.py::test_timeout_agotado` · `test_timeout_recupera_en_el_segundo_intento` · `test_clasificar_timeout_lanza_error_red`

### Salida truncada

**Mensaje:** `Salida truncada: el fragmento 0 supera 20000 caracteres y no se recorta en silencio.`

Si la base devuelve más fragmentos que el `k` pedido: `Salida truncada: la base devolvió 3 fragmentos y el limit/k pedido fue 2. No se recorta en silencio.` Si no devuelve una lista: `Salida truncada: similarity_search no devolvió la lista completa de fragmentos.` Un `finish_reason=length` se clasifica como `Salida truncada: la base cortó la respuesta. Detalle: finish_reason=length`.

**Cómo reproducirlo:** `db.respuesta = ["x" * 20001]` y `ainvoke` con `limit=3`. No se reintenta.

**Test:** `tests/test_errores.py::test_salida_truncada_no_recorta_ni_reintenta` · `test_mas_fragmentos_que_k_no_se_recorta` · `test_salida_no_es_lista` · `test_clasificar_truncado_no_es_transitorio`

### Error no transitorio

**Mensaje:** `Error de búsqueda: boom`

**Cómo reproducirlo:** encolar `RuntimeError("boom")`. Una sola llamada.

**Test:** `tests/test_errores.py::test_error_no_transitorio_no_reintenta`

## Evidencias

| Archivo | Qué muestra |
|---------|-------------|
| `evidencias/01-validacion-offline.txt` | `python validacion.py`: cadenas de `main.py` y la demo. |
| `evidencias/02-pytest.txt` | `python -m pytest -v`. |
| `evidencias/03-demo.txt` | `python main.py`. |

## Checklist

- [x] `SearchInput` con `query` (str) y `limit` (int), tipos estrictos, tope 1–10 y sin campos extra
- [x] `@tool(args_schema=SearchInput)` sobre `async def search_knowledge_base`
- [x] Docstring que dice cuándo usar la herramienta y cuándo no
- [x] `await db.similarity_search(query, k=limit)` — el `limit` de la llamada es el `k`
- [x] Default `limit=3` solo cuando la llamada no lo manda
- [x] Error de la herramienta devuelto como string (401/key, 429/cuota, red/timeout, salida truncada, validación)
- [x] `con_reintentos` usado dentro de `search_knowledge_base` (429 y red sí; 401 y truncado no)
- [x] `ejecutar_tool_call` valida con Pydantic antes de tocar la base
- [x] Contrato nombre / descripción / esquemas en el docstring de `main.py`
- [x] `if __name__ == "__main__"` ejecutable
- [x] pytest + mocks (sin API real)
- [x] README con una subsección por familia de error
- [x] `evidencias/` con salida real de comandos
