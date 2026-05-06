# TenMiNaTor v2.0

**Framework de Deep Learning ultraligero con cuantización avanzada**

[![PyPI version](https://badge.fury.io/py/tenminator.svg)](https://pypi.org/project/tenminator/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Instalación

```bash
# Núcleo (solo numpy)
pip install tenminator

# Con entrenamiento (PyTorch + Transformers)
pip install tenminator[training]

# Con integración Unsloth
pip install tenminator[unsloth]

# Con API REST
pip install tenminator[api]

# Todo incluido
pip install tenminator[all]
```

---

## Cuantización — Formatos soportados

| Formato | Bits | Técnica | VRAM 7B | Calidad | Uso recomendado |
|---------|------|---------|---------|---------|-----------------|
| **Q8** | 8 | INT8 simétrico | ~8 GB | ★★★★★ | Producción, máxima calidad |
| **Q6** | 6 | INT6 por grupo | ~6 GB | ★★★★★ | Alta calidad, ahorro moderado |
| **Q5** | 5 | INT5 asimétrico | ~5 GB | ★★★★☆ | Equilibrio calidad/tamaño |
| **Q4_K_M** | 4 | INT4 asimétrico | ~4 GB | ★★★★☆ | **Recomendado** — uso general |
| **Q3_TurboQuant** | 3 | Hadamard + INT3 | ~3 GB | ★★★☆☆ | GPUs con 8 GB VRAM |
| **Q2_KIVI** | 2 | KV-cache + outliers | ~2 GB | ★★☆☆☆ | Edge devices |
| **Q1_BitNet** | 1 | Ternario {-1,0,1} | ~1 GB | ★☆☆☆☆ | Investigación, chips especializados |
| **NVFP4** | 4 | Float4 Blackwell | ~4 GB | ★★★★☆ | NVIDIA RTX 50xx / Vera Rubin |
| **ANT4** | 4 | Adaptive Numerical | ~4 GB | ★★★★★ | NVIDIA Vera Rubin (ANT hardware) |

---

## Uso rápido

### Cuantizar un modelo

```python
from tenminator import Quantizer, QuantConfig
import numpy as np

# Simular pesos de un modelo
weights = {"layer1.weight": np.random.randn(4096, 4096).astype(np.float32)}

# Cuantizar a Q4 (recomendado)
q = Quantizer(QuantConfig(bits=4))
quantized = q.quantize_model(weights)

# Ver estadísticas
for name, qt in quantized.items():
    print(f"{name}: {qt.format_name}, {qt.data.nbytes / 1024:.1f} KB")

# Medir error
error = q.measure_error(weights["layer1.weight"], quantized["layer1.weight"])
print(f"MSE: {error['mse']:.6f} | Cosine: {error['cosine_similarity']:.4f}")
```

### TurboQuant Q3 (Google, ICLR 2026)

```python
from tenminator.quantization.formats import quantize_q3_turbo, dequantize

weights = np.random.randn(2048, 2048).astype(np.float32)

# Cuantizar con rotación Hadamard
qt = quantize_q3_turbo(weights, group_size=32)
print(f"Compresión: {weights.nbytes / qt.data.nbytes:.1f}x")

# Reconstruir
reconstructed = dequantize(qt)
```

### Q1 BitNet (1-bit)

```python
from tenminator.quantization.formats import quantize_q1_bitnet, dequantize

# Cuantización ternaria: {-1, 0, 1}
qt = quantize_q1_bitnet(weights)
print(f"Formato: {qt.format_name}")
print(f"Valores únicos: {np.unique(qt.data)}")  # [-1, 0, 1]
```

### Exportar a GGUF

```python
from tenminator.quantization.export import GGUFExporter

exporter = GGUFExporter()
path = exporter.export(quantized_model, "/tmp/modelo.gguf", arch="llama")
print(f"GGUF guardado: {path}")
```

### Exportar para chip (Taalas / unikernel)

```python
from tenminator.quantization.export import ChipExporter, UnikernelExporter

# Para chip Taalas
chip = ChipExporter()
path = chip.export(quantized_model, "/tmp/modelo.chip.bin", target_chip="taalas-v1")
area = chip.estimate_silicon_area(quantized_model)
print(f"Área estimada: {area['estimated_area_mm2']:.2f} mm²")

# Para unikernel (NanoVMs / Unikraft / Firecracker)
uni = UnikernelExporter()
path = uni.export(quantized_model, "/tmp/modelo.uni.bin", runtime="nanovms")
```

---

## Entrenamiento

```python
from tenminator import TrainingController, TrainingConfig

config = TrainingConfig(
    model_name="mi-modelo",
    learning_rate=1e-4,
    batch_size=4,
    max_steps=1000,
)

controller = TrainingController(config)
controller.start()
```

---

## Integración con el ecosistema yoqer

### TERMINATORI (inferencia)

```python
from tenminator import TerminatoriBridge

bridge = TerminatoriBridge(base_url="http://localhost:8000")
response = bridge.chat("Explica la cuantización Q4")
print(response["content"])
```

### TerminaTodo (almacenamiento)

```python
from tenminator import TerminaTodoBridge

storage = TerminaTodoBridge(base_url="http://localhost:8001")
url = storage.upload_model("/tmp/modelo.gguf", "modelos/mi-modelo-q4.gguf")
```

### Unsloth (fine-tuning eficiente)

```python
from tenminator import UnslothBridge

bridge = UnslothBridge()
bridge.finetune(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    dataset="mi_dataset.jsonl",
    output_dir="./modelo_finetuned",
    max_steps=500,
)
```

### LangChain

```python
from tenminator import LangChainRunnable

llm = LangChainRunnable(base_url="http://localhost:8000")
result = llm.invoke("¿Qué es TenMiNaTor?")
```

---

## CLI

```bash
# Información del sistema
tenminator info

# Recomendar formato de cuantización
tenminator recommend --vram 16 --model-size 7 --quality high

# Cuantizar modelo
tenminator quantize --model modelo.safetensors --bits 4 --output modelo_q4.gguf

# Exportar para unikernel
tenminator export --model modelo_q4.bin --format unikernel --runtime nanovms
```

---

## Ecosistema yoqer

| Paquete | PyPI | Descripción |
|---------|------|-------------|
| **tenminator** | `pip install tenminator` | Esta librería — entrenamiento y cuantización |
| **terminatori** | `pip install terminatori` | Motor de inferencia con panel web |
| **terminatodo** | `pip install terminatodo` | Gestión de almacenamiento multi-cloud |
| **terminator** | `pip install terminator-yoqer` | Framework de IA avanzado |
| **teminaTor** | `pip install teminaTor` | Framework de IA ligero |

---

## Licencia

MIT © yoqer
