# 📊 ML Inference Dashboard — РГР по дисциплине «Машинное обучение и большие данные»

## Описание проекта

Интерактивное веб-приложение (дашборд) на **Streamlit** для инференса (вывода) обученных моделей машинного обучения и визуализации данных.

Приложение включает 2 задачи:
- **Регрессия:** предсказание цены автомобиля по характеристикам (`cars`)
- **Классификация:** диагностика диабета по медицинским показателям (`diabetes`)

## Структура дашборда

| Страница | Описание |
|---|---|
| 🧑‍💻 О разработчике | ФИО, группа, стек технологий |
| 📁 О наборе данных | Описание признаков, EDA, предобработка |
| 📈 Визуализации | 4+ типа графиков (Matplotlib, Seaborn) |
| 🔮 Предсказание | Инференс 6 моделей ML с ручным вводом или загрузкой CSV |

## Обученные модели ML

Для каждой задачи используется по **6 моделей**:

| Модель | Тип | Библиотека |
|---|---|---|
| ML1 | Классическая (Ridge / Logistic Regression) | scikit-learn |
| ML2 | Бустинг (XGBoost) | xgboost |
| ML3 | Продвинутый градиентный бустинг (CatBoost) | catboost |
| ML4 | Бэггинг (Bagging) | scikit-learn |
| ML5 | Стэкинг (Stacking) | scikit-learn |
| ML6 | Глубокая полносвязная нейронная сеть | TensorFlow/Keras |

## Метрики качества

### Регрессия (R²)
- **Ridge:** 0.8253
- **XGBoost:** 0.7983
- **CatBoost:** 0.8265
- **Bagging:** 0.7406
- **Stacking:** 0.8266
- **Neural Network:** 0.8247

### Классификация (F1 weighted)
- **Logistic Regression:** 0.7140
- **XGBoost:** 0.7859
- **CatBoost:** 0.8398
- **Bagging:** 0.8036
- **Stacking:** 0.7960
- **Neural Network:** 0.7408

## Технологии

- Python 3.13
- Streamlit 1.58
- scikit-learn
- XGBoost, CatBoost
- TensorFlow/Keras
- Pandas, NumPy
- Matplotlib, Seaborn

## Установка и запуск

```bash
# 1. Клонировать репозиторий
git clone <ваш-репозиторий>
cd ргр

# 2. Создать виртуальное окружение
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate   # Linux/Mac

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Сгенерировать данные и обучить модели
python generate_data.py
python train_models.py

# 5. Запустить дашборд
streamlit run app.py
```

Приложение будет доступно по адресу: **http://localhost:8501**

## Развёртывание на Streamlit Cloud

1. Загрузите репозиторий на GitHub
2. Перейдите на [streamlit.io/cloud](https://streamlit.io/cloud)
3. Нажмите **New app** → выберите репозиторий
4. Укажите **Main file path:** `app.py`
5. Нажмите **Deploy**

## Деплой-ссылка

- **Streamlit Cloud:** [будет доступна после деплоя]
- **GitHub:** [ссылка на репозиторий]

## Автор

**ФИО:** [ФИО студента]  
**Группа:** МО-221 (пример)  
**Тема РГР:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных
