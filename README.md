# Insect Recognition Project

Classificazione di insetti parassiti agricoli tramite **Hybrid Knowledge Distillation** sul dataset [IP102](https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset).

Il progetto trasferisce la conoscenza di un ensemble teacher ad alto costo computazionale verso uno student compatto, mantenendo buone prestazioni con un numero ridotto di parametri.

---

## Descrizione

Il sistema è composto da due fasi principali:

1. **Training dei modelli Teacher** — tre architetture pre-addestrate su ImageNet-21K vengono fine-tuned sul dataset IP102
2. **Knowledge Distillation ibrida** — MobileNetV3-Large (student) apprende contemporaneamente da:
   - **Soft targets** dell'ensemble teacher (ConvNeXt-Base + Swin-Base)
   - **Feature distillation** tramite un Group Convolution Mapping Layer
   - **Hard targets** (cross-entropy classica)

---


---

## Struttura della Repository

```
Insect_Recognition-Project/
│
├── insects-recognition-convnext.ipynb        # Training ConvNeXt-Base
├── insects-recognition-swin.ipynb            # Training Swin-Base
├── insects-recognition-efficientnetV2-S.ipynb # Training EfficientNetV2-S
├── insects-recognition-mobilenetv3-large.ipynb # Baseline MobileNetV3 (no KD)
└── hybrid-distillation.ipynb                 # Knowledge Distillation ibrida
```

---


## Composizione del Dataset

**IP102** — benchmark fine-grained per la classificazione di 102 specie di insetti parassiti agricoli.

| Split      | Immagini |
|------------|----------|
| Train      | 45.095   |
| Validation | 7.508    |
| Test       | 22.619   |
| **Totale** | **75.222** |

---



### Hardware utilizzato

- **Piattaforma:** Kaggle (GPU NVIDIA Tesla P100 16GB)
- **Limite:** 30 ore settimanali, 12 ore per sessione

---
