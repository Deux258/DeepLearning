# DeepLearning

## Tarea 3 - Diego Muñoz, Benjamín Argüelles, Ignacio Pastén

- [Particiones y SFT de Phi-3](tarea3/entrenamiento.ipynb)
- [Evaluación zero-shot, few-shot y SFT](tarea3/evaluacion.ipynb)
- [Prompts comunes y demostraciones few-shot](tarea3/prompts.json)

### Entorno

**VS Code local.** Abre la carpeta raíz `DeepLearning` y abre los `.ipynb` en el editor. El proyecto incluye la ruta de búsqueda para ROCm. Si VS Code estaba abierto al cambiarla, ejecuta **Developer: Reload Window** desde `Ctrl+Shift+P`. En cada cuaderno, elige **Select Kernel → Select Another Kernel → Python Environments → rocm (3.14.10)** o **Jupyter Kernels → Tarea 3 (GPU)**. La primera celda confirma el Python que ejecuta el cuaderno. [Selector de kernels de VS Code](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management)

**Windows con RX 6600.** Ejecuta una vez desde la raíz del repositorio. El entorno GPU usa Python 3.14.10 por compatibilidad con la rueda estable ROCm para `gfx1032`.

```powershell
$taskRuntime = Join-Path (Get-Location) 'tarea3/.runtime'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $taskRuntime 'python'
$env:UV_CACHE_DIR = Join-Path $taskRuntime 'uv-cache'
uvx --from uv@latest uv venv --python 3.14.10 tarea3/.runtime/rocm
uvx --from uv@latest uv pip install --python tarea3/.runtime/rocm/Scripts/python.exe --index https://stable.repo.amd.com/rocm/whl-next/ 'torch[device-gfx1032]' -r tarea3/requirements.txt
tarea3/.runtime/rocm/Scripts/python.exe -m ipykernel install --user --name tarea3-gpu --display-name 'Tarea 3 (GPU)'
```

**Otra GPU local.** Crea un entorno con el [PyTorch para tu sistema y GPU](https://pytorch.org/get-started/locally/). Desde ese entorno instala `python -m pip install -U -r tarea3/requirements.txt` y registra `python -m ipykernel install --user --name tarea3-gpu --display-name "Tarea 3 (GPU)"`. Selecciona ese kernel en ambos cuadernos.

**Colab.** Activa una GPU y copia los dos `.ipynb`, `comun.py` y `prompts.json` a una misma carpeta. Desde una celda temporal, instala `%pip install -U accelerate bitsandbytes peft transformers`, reinicia la sesión y ubícate en esa carpeta con `%cd /content/tarea3` (ajusta la ruta según dónde copiaste los archivos). Guarda `salidas/` en almacenamiento persistente.

### Ejecución

1. Ejecutar `entrenamiento.ipynb` para descargar HH-RLHF, preparar las particiones y guardar el adaptador SFT. Las primeras tres celdas de código bastan para generar las particiones.
2. Ejecutar `evaluacion.ipynb` para generar sobre el conjunto final. Cada configuración tiene su propia celda, contador de respuestas y CSV. Las celdas de zero-shot y few-shot funcionan antes de entrenar. La última requiere el adaptador SFT. Apagar el kernel que dejó un modelo en memoria, incluido el del navegador si lo usaste, antes de pasar a otro cuaderno en la RX 6600.
3. Para validar el entorno o calibrar la pauta, fijar `PILOTO = True` en ambos cuadernos. El piloto actualiza el adaptador una vez y evalúa ocho prompts. Sus salidas tienen rutas propias.

### Intercambio de resultados

- **Integrante 2.** Usar `salidas/sft/` y los JSONL `recompensa_train`, `recompensa_val` y `ppo` de `salidas/particiones/`. Cada fila identifica la fuente y la línea del corpus original. El split PPO trae solo el prompt.
- **Integrante 3.** Usar `prompts.json` con las mismas cuatro demostraciones, en orden, y unir los CSV de `salidas/respuestas/final/` por `prompt_id` y `configuracion`. Las columnas `vacia` y `limite_tokens` registran resultados que conviene revisar durante la anotación.

La revisión del modelo y la del dataset están fijadas en `tarea3/comun.py`. Guardar el resultado de `uv pip freeze` del entorno usado junto a las salidas concretas de la corrida para reproducirla.
