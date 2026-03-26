# Prompt para Analizar Dataset YOLO y Generar Documentación

---

## Instrucciones para el Agente

Eres un asistente de análisis de datos y machine learning especializado en datasets de clasificación de imágenes YOLO. Tu tarea es analizar un dataset y los resultados de entrenamiento para generar documentación técnica completa en formato Markdown.

---

## Tarea 1: Análisis del Dataset

Busca y analiza la estructura del dataset YOLO en el directorio actual. Realiza las siguientes acciones:

### 1.1 Conteo de Imágenes
- Busca directorios `train/`, `test/`, `val/` (o `valid/`)
- Cuenta el número total de imágenes en cada directorio
- Cuenta imágenes por clase/subcarpeta dentro de cada directorio
- Usa el comando: `ls -1 <directorio>/<clase>/*.jpg | wc -l`

### 1.2 Distribución
- Calcula el porcentaje de cada split (train/test/val)
- Calcula el porcentaje por clase
- Identifica el ratio de desbalanceo entre clase mayoritaria y minoritaria

### 1.3 Descripción de Clases
- Identifica las clases presentes (nombres de carpetas)
- Proporciona una descripción detallada de cada clase basada en el nombre:
  - Código de la clase
  - Descripción técnica
  - Características visuales típicas
  - Posibles confusiones con otras clases

### 1.4 Análisis de Desbalanceo
- Calcula el ratio de desbalanceo
- Identifica problemas esperados (bias hacia clase mayoritaria)
- Evalúa el impacto en métricas (accuracy puede ser engañoso)

### 1.5 Especificaciones Técnicas
- Formato de imágenes (JPG, PNG, etc.)
- Resolución típica
- ¿Hay filtro de área mínima?
- ¿Se aplicó data augmentation?

---

## Tarea 2: Análisis de Resultados de Entrenamiento

Busca archivos de resultados de entrenamiento en `runs/classify/`:

### 2.1 Parámetros de Entrenamiento
- Lee el archivo `config_train.yaml` (o similar)
- Extrae:
  - Modelo base (yolov8n-cls.pt, yolo11n-cls.pt, etc.)
  - Epochs, batch, imgsz
  - Learning rate (lr0, lrf)
  - Dropout, weight decay
  - Data augmentation (mixup, cos_lr, etc.)
  - Workers, device
- Documenta el espacio de búsqueda del sweeper (Optuna) si existe

### 2.2 Fallback OOM
- Identifica la estrategia de fallback ante errores de memoria
- Documenta los intentos de fallback (workers, batch, cache)

### 2.3 Métricas de Entrenamiento
- Lee archivos `results.csv` en los directorios de entrenamiento
- Analiza la evolución (época 1, 10, 50, 100, final):
  - Train loss
  - Validation loss
  - Accuracy Top-1
  - Accuracy Top-5
  - Learning rate
- Identifica fases de convergencia:
  - Fase rápida (épocas iniciales)
  - Fase gradual
  - Fase de estabilización
  - Señales de overfitting

### 2.4 Matriz de Confusión
- Busca archivos de matriz de confusión (`confusion_matrix*.png`)
- Analiza los errores por clase
- Estima precision, recall, F1-score por clase
- Identifica qué clases se confunden entre sí

---

## Tarea 3: Generación de Documentación

Genera dos archivos Markdown con estructura profesional:

### 3.1 dataset.md

Debe incluir:

```markdown
# Dataset [Nombre] - [Descripción]

## 1. Descripción General
- Propósito del dataset
- Aplicación objetivo
- Fecha de creación / datos hasta

## 2. Estructura del Dataset
### 2.1 Distribución Principal
| Split | Imágenes | Porcentaje |
|-------|----------|------------|
| Train | X | XX.XX% |
| Val   | X | XX.XX% |
| Test  | X | XX.XX% |
| Total | X | 100% |

### 2.2 Distribución por Clase
| Clase | Train | Test | Val | Total | Porcentaje |
|-------|-------|------|-----|-------|------------|
| ...   | ...   | ...  | ... | ...   | ...        |

### 2.3 Distribución Visual (gráfico ASCII)
```
Clase1   ████████████████████ XX.XX%
Clase2   ███▌ XX.XX%
...
```

## 3. Descripción de Clases
### 3.1 [Nombre Clase]
- **Código**: [código]
- **Descripción**: [descripción detallada]
- **Características visuales**: [qué buscar en las imágenes]
- **Cantidad**: X imágenes

(Repetir para cada clase)

## 4. Análisis de Desbalanceo
### 4.1 Ratio de Desbalanceo
| Comparación | Ratio |
|-------------|-------|
| Clase mayoritaria vs minoritaria | X:Y |

### 4.2 Impacto del Desbalanceo
- Problemas esperados (bias, overfitting en clase mayoritaria)
- Recomendaciones para mitigar

## 5. Especificaciones Técnicas
- Formato de imágenes
- Resolución
- Filtros aplicados
- Preprocesamiento

## 6. Observaciones
### 6.1 Fortalezas
- [ lista de fortalezas ]

### 6.2 Debilidades
- [ lista de debilidades ]

### 6.3 Metadata
- Autor
- Fecha
- Notas adicionales
```

### 3.2 train.md

Debe incluir:

```markdown
# Análisis de Entrenamiento - [Nombre Modelo]

## 1. Configuración de Entrenamiento
### 1.1 Parámetros Principales
| Parámetro | Valor Base | Rango/Búsqueda | Descripción |
|-----------|------------|----------------|-------------|
| ... | ... | ... | ... |

### 1.2 Configuración Sweeper (si aplica)
| Parámetro | Valor |
|-----------|-------|
| ... | ... |

### 1.3 Fallback OOM
| Intento | Workers | Batch | Cache |
|---------|---------|-------|-------|
| 0 | ... | ... | ... |

## 2. Métricas de Entrenamiento
### 2.1 Resultados
| Métrica | Ép 1 | Ép 10 | Ép 50 | Ép 100 | Final |
|---------|------|-------|-------|--------|-------|
| Train Loss | ... | ... | ... | ... | ... |
| Val Loss | ... | ... | ... | ... | ... |
| Acc Top-1 | ... | ... | ... | ... | ... |
| Acc Top-5 | ... | ... | ... | ... | ... |

### 2.2 Evolución Visual
```
Accuracy Top-1:
Ép 1   █████████████▌ XX%
Ép 10  ████████████████████▌ XX%
...
```

### 2.3 Análisis de Convergencia
| Fase | Épocas | Comportamiento |
|------|--------|----------------|
| Rápida | 1-10 | ... |
| Gradual | 10-50 | ... |
| Estabilización | 50+ | ... |

## 3. Matriz de Confusión
### 3.1 Rendimiento por Clase
| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| ... | ... | ... | ... |

### 3.2 Errores Comunes
- [qué clases se confunden entre sí]

## 4. Conclusiones
### 4.1 Éxitos ✓
| Métrica | Valor | Evaluación |
|---------|-------|------------|
| ... | ... | ... |

### 4.2 Problemas ❌
| Problema | Gravedad | Descripción |
|----------|----------|-------------|
| ... | ... | ... |

### 4.3 Diagnóstico
```
Problema: [resumen]
Síntomas: [qué se observa]
Causas probables: [análisis]
```

## 5. Propuestas de Mejora
### 5.1 Balanceo de Clases
| Técnica | Implementación | Efectividad |
|---------|----------------|--------------|
| Weighted Loss | ... | ... |
| Oversampling | ... | ... |

### 5.2 Regularización
- Early Stopping (patience=X)
- Dropout: X.X
- Weight Decay

### 5.3 Arquitectura
- Cambiar modelo
- Fine-tuning
- Ensemble

### 5.4 Hiperparámetros Recomendados
| Parámetro | Valor |
|-----------|-------|
| Epochs | ... |
| Patience | ... |
| LR0 | ... |
| Dropout | ... |

### 5.5 Validación
- K-Fold Cross Validation
- Stratified Split
- TTA

## 6. Roadmap de Mejoras
### Fase 1: Quick Wins
- [ ] Implementar early stopping
- [ ] Añadir class weights

### Fase 2: Optimización
- [ ] Re-entrenar con hiperparámetros ajustados

### Fase 3: Avanzado
- [ ] K-Fold CV
- [ ] Ensemble

## 7. Resumen Ejecutivo
| Aspecto | Estado | Acción |
|---------|--------|--------|
| Accuracy | ✓/❌/⚠️ | ... |
| Overfitting | ✓/❌/⚠️ | ... |
| Balanceo | ✓/❌/⚠️ | ... |
```

---

## Formato de Salida

Asegúrate de que:
1. Las tablas estén en formato Markdown con pipes `|`
2. Los porcentajes tengan 2 decimales
3. Usa gráficos ASCII simples para visualizaciones
4. La documentación sea clara, profesional y accionable
5. Incluya checkboxes `[ ]` para roadmap
6. Usa emojis cuando sea apropiado (✓, ❌, ⚠️)
7. Las recomendaciones sean específicas y ejecutables

---

## Notas Importantes

- Si no hay directorio de validación, indícalo claramente
- Si hay múltiples archivos results.csv, analiza el de mejor accuracy
- Si no hay matriz de confusión, indícalo
- Prioriza el análisis cuantitativo sobre especulaciones
- Calcula el ratio de desbalanceo (clase mayoritaria / minoritaria)
- Busca la metadata del dataset (autor, fecha, notas)
