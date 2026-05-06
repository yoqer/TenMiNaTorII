# Informe Técnico: Sistema de Control de Entrenamiento 10×12 en TenMiNaTor II

**Versión analizada:** TenMiNaTor II v2.0.0  
**Fecha:** Abril 2026  
**Propósito:** Análisis detallado del mecanismo 10×12, su comportamiento sobre textos, y guía de monitorización desde la app TenMiNaTor I (tenminator-web)

---

## 1. ¿Qué es el Sistema 10×12?

El sistema **10×12** es el mecanismo de control de entrenamiento central de TenMiNaTor. Su nombre proviene de dos parámetros que trabajan juntos:

| Parámetro | Valor por defecto | Significado |
|-----------|-------------------|-------------|
| `max_tokens` | 10 | Número máximo de tokens (palabras/identificadores) por paquete de procesamiento |
| `early_stop_patience` / `token_limit_iterations` | 12 | Número máximo de iteraciones sin mejora (o con límite de tokens) antes de detener el entrenamiento |

En términos simples: el modelo **procesa el texto en paquetes de 10 tokens** y, si durante **12 iteraciones consecutivas** no observa mejora en la pérdida, **se detiene automáticamente**. Esta combinación da nombre al sistema: **10 tokens × 12 iteraciones = 10×12**.

El diseño responde a una necesidad práctica: TenMiNaTor está pensado para dispositivos con recursos limitados. Procesar textos completos de golpe sería inviable; dividirlos en paquetes pequeños y detener el entrenamiento cuando deja de aprender evita el desperdicio de cómputo.

---

## 2. Los Tres Modos de Parada

El `TrainingController` implementa tres condiciones de parada, que se evalúan en orden en cada llamada a `should_continue()`:

```
┌─────────────────────────────────────────────────────────────┐
│               should_continue() — Árbol de decisión         │
├─────────────────────────────────────────────────────────────┤
│  1. ¿current_iteration >= max_iterations (69)?  → PARA      │
│  2. ¿iterations_without_improvement >= 12?      → PARA      │
│  3. ¿tokens_procesados >= max_tokens (10)                   │
│     Y current_iteration >= token_limit_iterations (12)?     │
│                                                  → PARA      │
│  En cualquier otro caso                          → CONTINÚA  │
└─────────────────────────────────────────────────────────────┘
```

| Modo | Parámetro clave | Cuándo para |
|------|----------------|-------------|
| **Límite de iteraciones** | `max_iterations=69` | Siempre: tras 69 iteraciones como máximo absoluto |
| **Early Stopping** | `early_stop_patience=12` | Cuando la pérdida no mejora en 12 iteraciones seguidas |
| **Límite de tokens** | `max_tokens=10` + `token_limit_iterations=12` | Cuando se procesan ≥10 tokens Y se llevan ≥12 iteraciones |

El modo que puede causar una **parada inesperada** es el **Early Stopping (modo 2)**: el modelo llega a 12 iteraciones sin que la pérdida baje más de `min_delta=1e-6`, y se detiene aunque no haya llegado a las 69 iteraciones máximas.

---


__________________________________

## 3. Cómo se Parte un Texto: Ejemplo Paso a Paso

### En Frase de ejemplo: 

> **"El modelo aprende a reconocer patrones en el texto de entrenamiento."**

Esta frase tiene **12 palabras**. Veamos exactamente qué hace TenMiNaTor con ella.

### Paso 1 — Tokenización por palabras (modo `word`)

El `TransformerRLM` usa tokenización por palabras por defecto (`tokenizer='word'`). Cada palabra se convierte en un índice entero consultando un vocabulario de hasta 50.000 entradas:

```
Frase original:
"El modelo aprende a reconocer patrones en el texto de entrenamiento."

Tokens (índices enteros):
[ 412, 8031, 2917, 45, 11204, 6732, 89, 412, 9103, 67, 3841, 1 ]
  El  modelo aprende  a reconocer patrones en  el  texto  de entren. [EOS]
   0      1       2  3        4        5   6   7      8   9       10   11
```

El índice `1` representa el token de fin de secuencia `[EOS]`. El índice `412` aparece dos veces porque "El" y "el" comparten entrada (vocabulario insensible a mayúsculas en el modo por defecto).

### Paso 2 — Segmentación en paquetes de 10 tokens

Con `max_tokens=10`, el sistema **no procesa los 12 tokens de golpe**. Los divide en ventanas de 10:

```
Paquete 1 (tokens 0–9):
[ 412, 8031, 2917, 45, 11204, 6732, 89, 412, 9103, 67 ]
  El  modelo aprende  a reconocer patrones en  el  texto  de

Paquete 2 (tokens 10–11, incompleto):
[ 3841, 1 ]
  entrenamiento. [EOS]
```

El paquete 2 tiene solo 2 tokens; en la implementación actual se **rellena con padding** (índice 0) hasta completar la longitud de secuencia esperada por el modelo.

### Paso 3 — Construcción del par (entrada, objetivo)

Para cada paquete, el objetivo es **predecir el siguiente token** (tarea de modelado de lenguaje causal). Se aplica un desplazamiento de 1 posición (`np.roll(..., -1)`):

```
Entrada  (input):   [ El,  modelo,  aprende,  a,  reconocer,  patrones,  en,  el,  texto,  de ]
Objetivo (target):  [ modelo, aprende, a, reconocer, patrones, en, el, texto, de, entrenamiento ]
                      ↑ cada token predice el siguiente
```

El modelo aprende que después de "El" viene "modelo", después de "modelo" viene "aprende", y así sucesivamente.

### Paso 4 — Forward pass a través del Transformer

El paquete de 10 tokens pasa por las capas del `TransformerRLM`:

```
[10 tokens] → Embedding (dim=512) → [10 × 512]
           → PositionalEncoding    → [10 × 512]
           → TransformerBlock ×6   → [10 × 512]  (atención multi-cabeza + FFN)
           → Linear de salida      → [10 × 50000] (logits sobre vocabulario)
           → CrossEntropyLoss vs objetivo → pérdida escalar
```

La **MultiHeadAttention** con 8 cabezas permite que cada token "mire" a los otros 9 tokens del paquete para construir su representación contextual. Con solo 10 tokens, la ventana de atención es pequeña pero suficiente para capturar dependencias locales.

### Paso 5 — Actualización del estado en el controlador

Tras cada paquete, se llama a `controller.update(loss, tokens_processed=10)`:

```python
# Dentro de update():
self.state.current_iteration += 1          # +1 iteración
self.state.total_tokens_processed += 10    # +10 tokens acumulados

if loss < self.state.best_loss - 1e-6:
    self.state.best_loss = loss
    self.state.iterations_without_improvement = 0   # reinicia contador
else:
    self.state.iterations_without_improvement += 1  # incrementa contador
```

---

## 4. Simulación Completa: Las 12 Iteraciones que Paran el Entrenamiento

Usando la frase de ejemplo, simulamos lo que ocurre cuando el modelo ya ha aprendido lo básico y la pérdida se estanca:

| Iteración | Pérdida | Mejor pérdida | Sin mejora | ¿Continúa? |
|-----------|---------|---------------|------------|------------|
| 1 | 2.4831 | 2.4831 | 0 | Sí |
| 2 | 2.1204 | 2.1204 | 0 | Sí |
| 3 | 1.8932 | 1.8932 | 0 | Sí |
| 4 | 1.7105 | 1.7105 | 0 | Sí |
| 5 | 1.6891 | 1.6891 | 0 | Sí |
| 6 | 1.6900 | 1.6891 | **1** | Sí |
| 7 | 1.6887 | 1.6887 | 0 | Sí |
| 8 | 1.6890 | 1.6887 | **1** | Sí |
| 9 | 1.6889 | 1.6887 | **2** | Sí |
| 10 | 1.6891 | 1.6887 | **3** | Sí |
| 11 | 1.6888 | 1.6887 | **4** | Sí |
| ... | ... | ... | ... | Sí |
| 19 | 1.6886 | 1.6886 | 0 | Sí ← mejora mínima |
| 20 | 1.6887 | 1.6886 | **1** | Sí |
| 21–31 | ~1.6886 | 1.6886 | **2–12** | Sí → **No** |
| **32** | — | — | **12** | **PARA** |

En la iteración 32, `iterations_without_improvement` llega a 12 y `should_continue()` devuelve `False`. El mensaje en consola es:

```
[TenMiNaTor] Early Stop: 12 iteraciones sin mejora
stop_reason: "[NEW] Early Stop: 12 iteraciones sin mejora"
```

**Este comportamiento podria originar rápida parada al entrenar**: no es un error, es el Early Stopping funcionando correctamente. El modelo ha **convergido** y no tiene sentido seguir iterando.

---

## 5. ¿Cuántas Veces Entrena Cada Token?

Con la configuración por defecto y la frase de ejemplo:

- **Paquete 1** (10 tokens): se presenta al modelo en cada iteración hasta que el Early Stopping actúa. Si para en la iteración 32, ese paquete se ha visto **32 veces**.
- **Paquete 2** (2 tokens + padding): ídem, 32 presentaciones.
- **Total de tokens procesados**: `32 iteraciones × 10 tokens/iter = 320 tokens` (contando repeticiones).

En un dataset real con miles de frases, el dataloader barajea los paquetes en cada época, por lo que cada paquete se ve aproximadamente `max_iterations / num_batches` veces. Con 1.000 muestras, 32 batches de 32 y 69 iteraciones máximas, cada paquete se vería unas **2 veces** antes de que el Early Stopping o el límite de iteraciones actúe.

---

## 6. Por Qué Se Para: Diagnóstico del Problema

La parada prematura tiene tres causas posibles, ordenadas de más a menos probable:

**Causa A — Early Stopping demasiado agresivo (la más común).** Con `early_stop_patience=12`, si el dataset es pequeño o el learning rate es alto, la pérdida oscila sin bajar más de `1e-6` y el contador llega a 12 rápidamente. **Solución para TenMiNaTor III:** aumentar `early_stop_patience` a 20–30, o añadir un umbral relativo (`min_delta` como porcentaje de la pérdida actual en lugar de valor absoluto).

**Causa B — Límite de tokens alcanzado.** Si `max_tokens=10` y el dataset acumula rápidamente 10 tokens procesados (con batches pequeños), la condición 3 se activa antes de las 12 iteraciones. **Solución:** desactivar `max_tokens` (dejarlo en `None`) para datasets de texto libre.

**Causa C — Pérdida NaN o infinita.** Si el learning rate es demasiado alto, la pérdida explota a `inf` o `NaN`. Como `NaN > cualquier_número` es `False` en NumPy, `NaN < best_loss` también es `False`, por lo que el contador de "sin mejora" sube indefinidamente hasta llegar a 12. **Solución para TenMiNaTor III:** añadir validación `if not np.isfinite(loss): raise TrainingError(...)` en `update()`.

---

## 7. Cómo Monitorizar desde la App TenMiNaTor I (tenminator-web)

La aplicación web **tenminator-web** (TenMiNaTor I) expone el estado del controlador a través de su endpoint tRPC `training.getStatus`. Los campos relevantes del `TrainingState` que se pueden visualizar son:

| Campo en `TrainingState` | Qué muestra en la app | Dónde verlo |
|--------------------------|----------------------|-------------|
| `current_iteration` | Iteración actual / máximas | Barra de progreso en Training.tsx |
| `best_loss` | Mejor pérdida alcanzada | Panel de métricas |
| `iterations_without_improvement` | Contador Early Stopping | Badge numérico junto a la barra |
| `loss_history` | Historial de pérdidas | Gráfico LossChart (SVG en tiempo real) |
| `stop_reason` | Razón de parada | Alerta de estado al finalizar |
| `total_tokens_processed` | Tokens procesados | Estadística secundaria |
| `start_time` / `last_update_time` | Tiempo transcurrido | Reloj de sesión |

### Flujo de monitorización en tiempo real

La app realiza polling cada 2 segundos al endpoint `training.getStatus`. El ciclo visual es el siguiente:

```
Usuario inicia entrenamiento
        ↓
[Training.tsx] → trpc.training.start.useMutation()
        ↓
Servidor inicia TrainingController en proceso hijo
        ↓
[Polling cada 2s] → trpc.training.getStatus.useQuery()
        ↓
LossChart actualiza curva SVG con loss_history[]
StatusBadge muestra: "running" → "completed" / "failed"
        ↓
Cuando stop_reason != null → alerta con el motivo
```

### Cómo leer el contador 10×12 en la app

En la interfaz actual, el campo `iterations_without_improvement` se muestra en el panel de métricas como **"Sin mejora: X/12"**. Cuando este contador llega a 12, la barra de estado cambia a `completed` y aparece el mensaje:

```
[NEW] Early Stop: 12 iteraciones sin mejora
```

Para ver el historial completo de lo que ocurrió en una sesión anterior, la app guarda el `TrainingState` en el checkpoint `.pkl`. Se puede recargar con:

```python
from tenminator.training import TrainingController, TrainingConfig

config = TrainingConfig()
controller = TrainingController(None, None, None, config)
data = controller.load_checkpoint("./checkpoints/model_final.pkl")

print(controller.get_summary())
# Muestra: iteraciones, mejor pérdida, razón de parada, timestamps
```

### Visualización del historial en la app

El array `loss_history` del checkpoint contiene una pérdida por iteración. La app lo grafica con el componente `LossChart`. Para **reproducir** una sesión pasada en la app, se puede cargar el checkpoint y enviar el `loss_history` al endpoint `training.replay` (pendiente de implementar en TenMiNaTor III).

---

## 8. Resumen Ejecutivo y Recomendaciones para TenMiNaTor III

El sistema 10×12 es funcionalmente correcto pero tiene tres limitaciones que pueden corregirse en la versión III:

**Limitación 1 — `min_delta` absoluto.** El umbral de mejora `1e-6` es demasiado estricto para pérdidas grandes (>1.0). Una pérdida que pasa de 2.4831 a 2.4830 es una mejora real pero no supera `1e-6`. **Propuesta:** usar `min_delta_rel = 0.001` (0,1% de mejora relativa).

**Limitación 2 — Sin distinción entre paquetes.** El contador `iterations_without_improvement` es global: si el modelo mejora en el paquete 1 pero no en el paquete 2, el contador sube igualmente. **Propuesta:** contador por paquete (`per_chunk_patience`).

**Limitación 3 — Sin validación de NaN.** La pérdida NaN no se detecta y provoca una parada silenciosa. **Propuesta:** añadir `assert np.isfinite(loss), f"Pérdida no finita: {loss}"` en `update()`.

| Mejora propuesta | Impacto | Prioridad |
|-----------------|---------|-----------|
| `min_delta_rel` en lugar de absoluto | Evita paradas prematuras en pérdidas altas | Alta |
| Validación de NaN/Inf en `update()` | Evita paradas silenciosas | Alta |
| Contador de paciencia por paquete | Entrenamiento más granular | Media |
| Endpoint `training.replay` en la app | Monitorización de sesiones pasadas | Media |
| Aumentar `early_stop_patience` por defecto a 20 | Más tolerancia en datasets pequeños | Baja |

---

*Informe generado a partir del análisis del código fuente de TenMiNaTor II v2.0.0: `training.py`, `nn.py`, `ARCHITECTURAL_DESIGN.md`, `test_10x12_system.py` y `train_complete_model.py`.*
