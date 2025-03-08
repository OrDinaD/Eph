# Emotion Voice Converter

Система для преобразования эмоциональной окраски голоса с сохранением характеристик исходного голоса. Проект использует глубокое обучение для изменения эмоциональной составляющей речи, сохраняя при этом идентичность говорящего и содержание сказанного.

## Особенности

- Преобразование аудио файлов с одной эмоцией в другую
- Сохранение характеристик исходного голоса
- Поддержка различных эмоциональных состояний
- Модульная архитектура с возможностью расширения
- Интеграция с современными вокодерами
- Поддержка различных форматов аудио (конвертация из OGG в WAV)

## Требования

- Python 3.8+
- PyTorch 2.0+
- CUDA (опционально, для ускорения на GPU)
- FFmpeg (для обработки аудио)

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/emotion-voice-converter.git
cd emotion-voice-converter
```

2. Создайте виртуальное окружение и активируйте его:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Структура проекта

```
emotion-voice-converter/
├── src/
│   ├── data/           # Модули для работы с данными
│   ├── models/         # Архитектура нейронной сети
│   ├── preprocessing/  # Обработка аудио
│   ├── training/       # Обучение модели
│   ├── utils/         # Вспомогательные функции
│   └── vocoder/       # Интеграция с вокодером
├── configs/           # Конфигурационные файлы
├── tests/            # Тесты
├── data/             # Датасет (не включен в репозиторий)
│   ├── raw/          # Исходные аудио файлы
│   └── processed/    # Обработанные файлы
└── checkpoints/      # Сохраненные модели
```

## Подготовка данных

1. Подготовьте датасет в следующем формате:
   - Аудио файлы в формате WAV или OGG
   - CSV файл с метаданными, содержащий колонки:
     - file_path: путь к аудио файлу
     - emotion: метка эмоции
     - transcription: текстовая транскрипция (опционально)

2. Разместите аудио файлы в директории `data/raw/`

3. Укажите путь к CSV файлу с метаданными при запуске обучения

## Обучение модели

Для запуска обучения используйте скрипт `src/train.py`:

```bash
python src/train.py \
    --config configs/config.py \
    --data_dir data/raw \
    --metadata data/metadata.csv \
    [--checkpoint checkpoints/previous_model.pt]
```

## Инференс

Для преобразования аудио используйте скрипт inference.py:

```bash
python src/inference.py \
    --input path/to/input.wav \
    --output path/to/output.wav \
    --emotion "happy" \
    --model checkpoints/best_model.pt
```

## Мониторинг обучения

Проект использует Weights & Biases для мониторинга процесса обучения. Метрики включают:
- Ошибку реконструкции
- Ошибку цикличности
- Точность классификации эмоций
- Примеры преобразованного аудио

## Оценка качества

Для оценки качества преобразования используются следующие метрики:
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- MOS (Mean Opinion Score) через субъективное тестирование

## Лицензия

MIT

## Цитирование

Если вы используете этот проект в своих исследованиях, пожалуйста, процитируйте его:

```bibtex
@software{emotion_voice_converter,
  author = {Your Name},
  title = {Emotion Voice Converter},
  year = {2024},
  url = {https://github.com/yourusername/emotion-voice-converter}
}
``` 