# Industrial Defect Inspection

Проект компьютерного зрения для автоматического определения дефектов промышленных деталей.

Модель классифицирует изображения головок винтов на два класса:

- `NG` — дефектная деталь
- `OK` — нормальная деталь

Проект написан на Python с использованием PyTorch.

## Current results

Модель была обучена на M2 Screw Head Defect Dataset.

Результаты на тестовой выборке:

| Metric | Result |
|---|---:|
| Accuracy | 88.06% |
| Precision NG | 98.64% |
| Recall NG | 78.14% |
| F1 NG | 87.20% |
| Recall OK | 98.83% |

Высокий Precision NG означает, что если модель определяет деталь как дефектную, она почти всегда действительно является дефектной.

При этом Recall NG показывает, что модель пока пропускает часть дефектов. Это одно из направлений дальнейшего улучшения модели.

## Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

## Project structure

```text
industrial-defect-inspection/
├── model.py
├── train.py
├── evaluate.py
├── models/
│   └── defect_cnn.pth
├── assets/
│   └── confusion_matrix.png
├── requirements.txt
├── README.md
└── .gitignore
```

`model.py` содержит архитектуру CNN.

`train.py` загружает обучающий датасет, обучает модель на GPU и сохраняет веса.

`evaluate.py` загружает сохранённую модель и оценивает её на тестовом наборе данных.

## Model architecture

Модель представляет собой простую CNN:

```text
RGB image 128x128
        ↓
Conv2d 3 → 16
        ↓
ReLU
        ↓
MaxPool
        ↓
Conv2d 16 → 32
        ↓
ReLU
        ↓
MaxPool
        ↓
Flatten
        ↓
Linear
        ↓
NG / OK
```

Для обучения используется:

- CrossEntropyLoss
- Adam optimizer
- batch size 32
- CUDA, если доступна NVIDIA GPU

## Dataset

Используется M2 Screw Head Defect Dataset.

Датасет содержит изображения нормальных и дефектных головок винтов.

Сам датасет не хранится в репозитории.

Ожидаемая структура:

```text
HEAD/
├── Train/
│   ├── NG/
│   └── OK/
└── Test/
    ├── NG/
    └── OK/
```

## Training

Запуск обучения:

```bash
python train.py
```

После обучения веса сохраняются:

```text
models/defect_cnn.pth
```

## Evaluation

Для оценки обученной модели:

```bash
python evaluate.py
```

Скрипт рассчитывает:

- Accuracy
- Precision для дефектных деталей
- Recall
- F1-score
- Confusion Matrix

## Roadmap

В следующих версиях планируется добавить:

- inference для отдельного изображения
- вероятность предсказания
- улучшенную CNN
- график обучения
- anomaly detection
- локализацию дефекта
- heatmap дефектной области
- возможный перенос облегчённой модели на ESP32-S3