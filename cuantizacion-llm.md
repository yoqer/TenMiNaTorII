# Cuantización de modelos LLM — Referencia técnica

## Formatos soportados por TenMiNaTor

| Formato | Bits | Técnica | VRAM 7B | Calidad | Caso de uso |
|---------|------|---------|---------|---------|-------------|
| Q8 | 8 | INT8 simétrico por grupo | ~8 GB | ★★★★★ | Producción máxima calidad |
| Q6 | 6 | INT6 por grupo | ~6 GB | ★★★★★ | Alta calidad, ahorro moderado |
| Q5 | 5 | INT5 asimétrico | ~5 GB | ★★★★☆ | Equilibrio calidad/tamaño |
| Q4_K_M | 4 | INT4 asimétrico (recomendado) | ~4 GB | ★★★★☆ | Uso general |
| Q3_TurboQuant | 3 | Rotación Hadamard + INT3 | ~3 GB | ★★★☆☆ | GPUs 8 GB VRAM |
| Q2_KIVVI | 2 | KV-cache + outliers | ~2 GB | ★★☆☆☆ | Edge devices |
| Q1_BitNet | 1 | Ternario {-1,0,1} | ~1 GB | ★☆☆☆☆ | Chips especializados |
| NVFP4 | 4 | Float4 con codebook (Blackwell) | ~4 GB | ★★★★☆ | NVIDIA RTX 50xx |
| ANT4 | 4 | Adaptive Numerical Type | ~4 GB | ★★★★★ | NVIDIA Vera Rubin |

## Estructura de módulo de cuantización

```
tenminator/
└── quantization/
    ├── __init__.py      # Exporta Quantizer, QuantConfig, dequantize
    ├── engine.py        # Clase Quantizer + QuantConfig (Pydantic)
    ├── formats.py       # Funciones quantize_q1..q8, quantize_nvfp4, quantize_ant4
    └── export.py        # GGUFExporter, ChipExporter, UnikernelExporter
```

## Patrón de implementación: formats.py

```python
import numpy as np
from dataclasses import dataclass
from typing import Literal

@dataclass
class QuantizedTensor:
    data: np.ndarray          # datos cuantizados
    scale: np.ndarray         # escala por grupo
    zero_point: np.ndarray    # punto cero (asimétrico)
    bits: int                 # 1, 2, 3, 4, 5, 6, 8
    format_name: str          # "Q4_K_M", "Q3_TurboQuant", etc.
    group_size: int           # 32, 64, 128
    metadata: dict            # info adicional

def quantize_q4(weights: np.ndarray, group_size: int = 32) -> QuantizedTensor:
    """INT4 asimétrico por grupo — recomendado para uso general."""
    w = weights.reshape(-1, group_size)
    w_min = w.min(axis=1, keepdims=True)
    w_max = w.max(axis=1, keepdims=True)
    scale = (w_max - w_min) / 15.0
    scale = np.where(scale == 0, 1e-8, scale)
    zero_point = np.round(-w_min / scale).clip(0, 15).astype(np.uint8)
    q = np.round(w / scale + zero_point).clip(0, 15).astype(np.uint8)
    return QuantizedTensor(
        data=q.reshape(weights.shape), scale=scale.reshape(-1),
        zero_point=zero_point.reshape(-1), bits=4,
        format_name="Q4_K_M", group_size=group_size, metadata={}
    )

def quantize_q3_turbo(weights: np.ndarray, group_size: int = 32) -> QuantizedTensor:
    """TurboQuant Q3: rotación Hadamard + INT3 (Google ICLR 2026)."""
    # Rotación Hadamard para distribuir outliers
    n = weights.shape[-1]
    h_size = 1 << (n - 1).bit_length()
    padded = np.pad(weights.reshape(-1), (0, h_size - weights.size % h_size or 0))
    # Aplicar transformada Hadamard simplificada
    h = padded.reshape(-1, h_size)
    for i in range(int(np.log2(h_size))):
        step = 1 << i
        h_new = h.copy()
        h_new[:, ::2*step] = (h[:, ::2*step] + h[:, step::2*step]) / np.sqrt(2)
        h_new[:, step::2*step] = (h[:, ::2*step] - h[:, step::2*step]) / np.sqrt(2)
        h = h_new
    rotated = h.flatten()[:weights.size].reshape(weights.shape)
    # Cuantizar a 3 bits
    w = rotated.reshape(-1, group_size)
    w_min = w.min(axis=1, keepdims=True)
    w_max = w.max(axis=1, keepdims=True)
    scale = (w_max - w_min) / 7.0
    scale = np.where(scale == 0, 1e-8, scale)
    zero_point = np.round(-w_min / scale).clip(0, 7).astype(np.uint8)
    q = np.round(w / scale + zero_point).clip(0, 7).astype(np.uint8)
    return QuantizedTensor(
        data=q.reshape(weights.shape), scale=scale.reshape(-1),
        zero_point=zero_point.reshape(-1), bits=3,
        format_name="Q3_TurboQuant", group_size=group_size,
        metadata={"hadamard_applied": True}
    )

def quantize_q1_bitnet(weights: np.ndarray) -> QuantizedTensor:
    """BitNet 1-bit: ternario {-1, 0, 1} empaquetado."""
    threshold = np.abs(weights).mean() * 0.5
    q = np.where(weights > threshold, 1,
        np.where(weights < -threshold, -1, 0)).astype(np.int8)
    scale = np.abs(weights).mean(keepdims=True)
    return QuantizedTensor(
        data=q, scale=scale, zero_point=np.zeros(1),
        bits=1, format_name="Q1_BitNet", group_size=weights.size,
        metadata={"threshold": float(threshold), "sparsity": float((q == 0).mean())}
    )

def dequantize(qt: QuantizedTensor) -> np.ndarray:
    """Reconstruir tensor float32 desde cualquier formato cuantizado."""
    if qt.bits == 1:
        return qt.data.astype(np.float32) * qt.scale
    g = qt.group_size
    q = qt.data.reshape(-1, g).astype(np.float32)
    s = qt.scale.reshape(-1, 1)
    z = qt.zero_point.reshape(-1, 1).astype(np.float32)
    return ((q - z) * s).reshape(qt.data.shape)
```

## Patrón de implementación: engine.py

```python
from pydantic import BaseModel
from typing import Dict, Optional, Literal
import numpy as np

class QuantConfig(BaseModel):
    bits: int = 4
    group_size: int = 32
    format: str = "auto"  # "auto" elige según bits
    mixed_precision: bool = False

class Quantizer:
    FORMATS = {
        1: "Q1_BitNet", 2: "Q2_KIVI", 3: "Q3_TurboQuant",
        4: "Q4_K_M", 5: "Q5", 6: "Q6", 8: "Q8",
    }
    
    def __init__(self, config: QuantConfig = None):
        self.config = config or QuantConfig()
    
    def quantize_model(self, weights: Dict[str, np.ndarray]) -> Dict[str, QuantizedTensor]:
        from .formats import quantize_q4, quantize_q3_turbo, quantize_q1_bitnet
        result = {}
        for name, w in weights.items():
            if self.config.bits == 4:
                result[name] = quantize_q4(w, self.config.group_size)
            elif self.config.bits == 3:
                result[name] = quantize_q3_turbo(w, self.config.group_size)
            elif self.config.bits == 1:
                result[name] = quantize_q1_bitnet(w)
        return result
    
    def measure_error(self, original: np.ndarray, qt: QuantizedTensor) -> dict:
        from .formats import dequantize
        reconstructed = dequantize(qt)
        mse = float(np.mean((original - reconstructed) ** 2))
        cos = float(np.dot(original.flatten(), reconstructed.flatten()) /
                   (np.linalg.norm(original) * np.linalg.norm(reconstructed) + 1e-8))
        return {"mse": mse, "cosine_similarity": cos, "bits": qt.bits}
    
    def recommend(self, vram_gb: float, model_size_b: float, quality: str = "balanced") -> str:
        """Recomendar formato según VRAM disponible y tamaño del modelo."""
        vram_per_param = vram_gb / model_size_b
        if vram_per_param >= 2.0: return "Q8"
        if vram_per_param >= 1.0: return "Q6"
        if vram_per_param >= 0.7: return "Q5"
        if vram_per_param >= 0.6: return "Q4_K_M"
        if vram_per_param >= 0.45: return "Q3_TurboQuant"
        if vram_per_param >= 0.3: return "Q2_KIVI"
        return "Q1_BitNet"
```

## Patrón de implementación: export.py

```python
import struct, os, json
from pathlib import Path

class GGUFExporter:
    """Exportar a formato GGUF para llama.cpp / Ollama."""
    MAGIC = b"GGUF"
    VERSION = 3
    
    def export(self, model: dict, path: str, arch: str = "llama") -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<I", self.VERSION))
            f.write(struct.pack("<Q", len(model)))  # n_tensors
            f.write(struct.pack("<Q", 2))            # n_kv (metadata entries)
            # Metadata: arquitectura y tipo de cuantización
            self._write_kv(f, "general.architecture", arch)
            bits = next(iter(model.values())).bits if model else 4
            self._write_kv(f, "general.quantization_version", f"Q{bits}")
            # Tensores
            for name, qt in model.items():
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", qt.bits))
                f.write(struct.pack("<Q", qt.data.nbytes))
                f.write(qt.data.tobytes())
        return str(path)
    
    def _write_kv(self, f, key: str, value: str):
        k = key.encode("utf-8")
        v = value.encode("utf-8")
        f.write(struct.pack("<I", len(k))); f.write(k)
        f.write(struct.pack("<I", 8))  # tipo string
        f.write(struct.pack("<Q", len(v))); f.write(v)

class ChipExporter:
    """Exportar para chips Taalas y hardware especializado."""
    def export(self, model: dict, path: str, target_chip: str = "taalas-v1") -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "format": "CHIP_BIN_v1",
            "target": target_chip,
            "layers": len(model),
            "total_bytes": sum(qt.data.nbytes for qt in model.values()),
        }
        with open(path, "wb") as f:
            h = json.dumps(header).encode("utf-8")
            f.write(struct.pack("<I", len(h))); f.write(h)
            for name, qt in model.items():
                n = name.encode("utf-8")
                f.write(struct.pack("<I", len(n))); f.write(n)
                f.write(struct.pack("<I", qt.bits))
                f.write(struct.pack("<Q", qt.data.nbytes))
                f.write(qt.data.tobytes())
        return str(path)
    
    def estimate_silicon_area(self, model: dict) -> dict:
        total_params = sum(qt.data.size for qt in model.values())
        bits = next(iter(model.values())).bits if model else 4
        area_mm2 = (total_params * bits) / (8 * 1e6 * 10)  # ~10M bits/mm² @ 5nm
        return {"total_params": total_params, "bits": bits,
                "estimated_area_mm2": round(area_mm2, 4)}

class UnikernelExporter:
    """Exportar para NanoVMs, Unikraft y Firecracker."""
    RUNTIMES = ["nanovms", "unikraft", "firecracker", "ops"]
    
    def export(self, model: dict, path: str, runtime: str = "nanovms") -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "format": "UNIKERNEL_v1",
            "runtime": runtime,
            "memory_mb": sum(qt.data.nbytes for qt in model.values()) // (1024*1024) + 64,
            "layers": len(model),
        }
        with open(path, "wb") as f:
            h = json.dumps(header).encode("utf-8")
            f.write(struct.pack("<I", len(h))); f.write(h)
            for name, qt in model.items():
                n = name.encode("utf-8")
                f.write(struct.pack("<I", len(n))); f.write(n)
                f.write(struct.pack("<Q", qt.data.nbytes))
                f.write(qt.data.tobytes())
        return str(path)
```

## Tests mínimos para cuantización

```python
import pytest, numpy as np

@pytest.fixture
def sample_weights():
    np.random.seed(42)
    return np.random.randn(256, 256).astype(np.float32)

class TestQuantizer:
    def test_q4_roundtrip(self, sample_weights):
        from tenminator.quantization.formats import quantize_q4, dequantize
        qt = quantize_q4(sample_weights)
        rec = dequantize(qt)
        mse = float(np.mean((sample_weights - rec) ** 2))
        assert mse < 0.05, f"Q4 MSE too high: {mse}"
    
    def test_q3_turbo(self, sample_weights):
        from tenminator.quantization.formats import quantize_q3_turbo
        qt = quantize_q3_turbo(sample_weights)
        assert qt.bits == 3
        assert qt.metadata.get("hadamard_applied") is True
    
    def test_q1_bitnet_values(self, sample_weights):
        from tenminator.quantization.formats import quantize_q1_bitnet
        qt = quantize_q1_bitnet(sample_weights)
        assert set(np.unique(qt.data)).issubset({-1, 0, 1})
    
    def test_gguf_export(self, sample_weights, tmp_path):
        from tenminator.quantization.formats import quantize_q4
        from tenminator.quantization.export import GGUFExporter
        model = {"layer.weight": quantize_q4(sample_weights)}
        path = GGUFExporter().export(model, str(tmp_path / "model.gguf"))
        assert Path(path).exists()
        assert Path(path).read_bytes()[:4] == b"GGUF"
    
    def test_recommend(self):
        from tenminator.quantization.engine import Quantizer
        q = Quantizer()
        assert q.recommend(vram_gb=4, model_size_b=7) in ["Q3_TurboQuant", "Q4_K_M"]
        assert q.recommend(vram_gb=32, model_size_b=7) == "Q8"
```

## Tabla de VRAM recomendada por modelo y formato

| Modelo | Q8 | Q6 | Q5 | Q4 | Q3 | Q2 | Q1 |
|--------|----|----|----|----|----|----|-----|
| 1B | 1 GB | 0.8 GB | 0.6 GB | 0.5 GB | 0.4 GB | 0.3 GB | 0.2 GB |
| 3B | 3 GB | 2.3 GB | 1.9 GB | 1.5 GB | 1.1 GB | 0.8 GB | 0.4 GB |
| 7B | 8 GB | 6 GB | 5 GB | 4 GB | 3 GB | 2 GB | 1 GB |
| 13B | 14 GB | 10 GB | 9 GB | 7 GB | 5 GB | 4 GB | 2 GB |
| 30B | 32 GB | 24 GB | 20 GB | 16 GB | 12 GB | 8 GB | 4 GB |
| 40B | 42 GB | 32 GB | 26 GB | 20 GB | 15 GB | 10 GB | 5 GB |
| 70B | 70 GB | 52 GB | 44 GB | 35 GB | 26 GB | 18 GB | 9 GB |

## Notas sobre tecnologías de despliegue

**Taalas chips:** Chips de inferencia especializados para LLMs cuantizados. El formato `.chip.bin` incluye cabecera JSON + tensores en binario. El chip lee directamente sin conversión adicional.

**NanoVMs (OPS):** Unikernel para Linux. Comando de despliegue:
```bash
ops load modelo.uni.bin -p 8080 --config ops.json
```

**Unikraft:** Framework de unikernel modular. Requiere compilar con `kraft build` y el `.uni.bin` como payload.

**Firecracker:** MicroVM de AWS. El `.uni.bin` se monta como rootfs en una microVM con 256 MB RAM mínimo.

**NVFP4 / ANT4 (NVIDIA Vera Rubin):** Requieren driver CUDA 13+ y GPU Blackwell (RTX 50xx) o Vera Rubin. El formato usa codebook de 16 valores float4 por grupo de 16 pesos.
