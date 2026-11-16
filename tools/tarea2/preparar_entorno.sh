#!/usr/bin/env bash
# Ejecutar desde Linux: bash tools/tarea2/preparar_entorno.sh
# PyTorch CPU: https://pytorch.org/get-started/locally/
set -euo pipefail

TAREA2_PROYECTO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TAREA2_PROYECTO"

if [[ ! -x .venv/bin/python ]]; then
    "${TAREA2_PYTHON:-python3}" -m venv .venv
fi

printf '%s\n' 'Instalando PyTorch y Torchvision para CPU. Este paso requiere acceso a Internet.'
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

printf '%s\n' 'Instalando las dependencias del notebook.'
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check

printf '%s\n' 'Comprobando las importaciones y una operación de detección.'
.venv/bin/python - <<'PY'
import importlib.metadata
import sys

import ipykernel
import matplotlib.pyplot as plt
import numpy
import PIL
import pycocotools.coco
import torch
import torchvision
import ultralytics
import yaml
from torchvision.ops import nms

assert importlib.metadata.version('ultralytics') == '8.4.174'
indices = nms(torch.tensor([[0., 0., 2., 2.]]), torch.tensor([1.]), .5)
assert indices.tolist() == [0]
print('Python:', sys.executable)
print('PyTorch:', torch.__version__, '| Torchvision:', torchvision.__version__)
print('Importaciones y NMS en CPU: correctos.')
PY

.venv/bin/python -m ipykernel install --user --name deeplearning-tarea2 \
    --display-name 'DeepLearning - Tarea 2 (.venv, CPU)'
printf '%s\n' 'Entorno listo. En el notebook selecciona: DeepLearning - Tarea 2 (.venv, CPU).'
