import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from catboost import CatBoostRegressor, CatBoostClassifier
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
    print("TensorFlow/Keras доступен")
except ImportError as e:
    KERAS_AVAILABLE = False
    keras = None
    print(f"TensorFlow/Keras не доступен: {e}")
import warnings
warnings.filterwarnings('ignore')

st.title("🔮 Предсказание моделей ML")

task = st.selectbox("Выберите задачу:", [
    "Регрессия: Предсказание цены авто (cars)",
    "Классификация: Диагностика диабета (diabetes)"
])

is_regression = "Регрессия" in task

# ==== Определение моделей ====
if is_regression:
    MODELS = {
        "ML1 — Ridge Regression (sklearn)": "models_regression/ridge.pkl",
        "ML2 — XGBoost (Boosting)": "models_regression/xgboost.pkl",
        "ML3 — CatBoost": "models_regression/catboost.cbm",
        "ML4 — Bagging (sklearn)": "models_regression/bagging.pkl",
        "ML5 — Stacking (sklearn)": "models_regression/stacking.pkl",
    }
    # Всегда добавляем нейросеть, но помечаем если TensorFlow недоступен
    if KERAS_AVAILABLE:
        MODELS["ML6 — Neural Network (TensorFlow)"] = "models_regression/neural_network.keras"
    else:
        MODELS["ML6 — Neural Network (TensorFlow)"] = "models_regression/neural_network.keras"
else:
    MODELS = {
        "ML1 — Logistic Regression (sklearn)": "models_classification/logreg.pkl",
        "ML2 — XGBoost (Boosting)": "models_classification/xgboost.pkl",
        "ML3 — CatBoost": "models_classification/catboost.cbm",
        "ML4 — Bagging (sklearn)": "models_classification/bagging.pkl",
        "ML5 — Stacking (sklearn)": "models_classification/stacking.pkl",
    }
    # Всегда добавляем нейросеть, но помечаем если TensorFlow недоступен
    if KERAS_AVAILABLE:
        MODELS["ML6 — Neural Network (TensorFlow)"] = "models_classification/neural_network.keras"
    else:
        MODELS["ML6 — Neural Network (TensorFlow)"] = "models_classification/neural_network.keras"

model_name = st.selectbox("Выберите модель:", list(MODELS.keys()))

# ==== Загрузка модели ====
@st.cache_resource
def load_model(path, model_name):
    try:
        if path.endswith('.keras'):
            if not KERAS_AVAILABLE:
                st.error("❌ TensorFlow/Keras недоступен в данном окружении.")
                st.info("💡 Установите TensorFlow: `pip install tensorflow` или выберите другую модель.")
                st.stop()
            st.info(f"Загрузка нейросети: {model_name}")
            return keras.models.load_model(path)
        elif path.endswith('.cbm'):
            if is_regression:
                model = CatBoostRegressor()
            else:
                model = CatBoostClassifier()
            model.load_model(path)
            return model
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Ошибка загрузки модели {model_name}: {str(e)}")
        st.stop()

model_path = MODELS[model_name]
if os.path.exists(model_path):
    model = load_model(model_path, model_name)
    st.success(f"Модель загружена: {model_name}")
else:
    st.error(f"Модель не найдена: {model_path}")
    st.stop()

# ==== Предобработка ====
@st.cache_resource
def load_preprocessor(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ==== Ввод данных ====
input_method = st.radio("Способ ввода данных:", ["📝 Ручной ввод", "📁 Загрузка CSV"])

if is_regression:
    # ===== РЕГРЕССИЯ: ВВОД =====
    if input_method == "📝 Ручной ввод":
        st.subheader("Введите характеристики автомобиля")
        col1, col2 = st.columns(2)
        with col1:
            brand = st.selectbox("Марка", ['Toyota', 'Honda', 'BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Ford', 'Hyundai', 'Kia', 'Nissan'])
            model_car = st.selectbox("Модель", {'Toyota': ['Corolla', 'Camry', 'RAV4', 'Land Cruiser'],
                'Honda': ['Civic', 'Accord', 'CR-V'],
                'BMW': ['3 Series', '5 Series', 'X3', 'X5'],
                'Mercedes': ['C-Class', 'E-Class', 'GLC', 'GLE'],
                'Audi': ['A4', 'A6', 'Q5', 'Q7'],
                'Volkswagen': ['Golf', 'Passat', 'Tiguan', 'Touareg'],
                'Ford': ['Focus', 'Mondeo', 'Kuga'],
                'Hyundai': ['Solaris', 'Elantra', 'Tucson', 'Santa Fe'],
                'Kia': ['Rio', 'Cerato', 'Sportage', 'Sorento'],
                'Nissan': ['Almera', 'Teana', 'X-Trail', 'Patrol']
            }[brand])
            year = st.slider("Год выпуска", 2000, 2024, 2015)
            mileage = st.number_input("Пробег (км)", min_value=0, max_value=500000, value=100000, step=1000)
        with col2:
            engine = st.selectbox("Объём двигателя (л)", [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.5, 2.7, 3.0, 3.5, 4.0, 5.0])
            transmission = st.selectbox("КПП", ['Manual', 'Automatic', 'CVT', 'Robot'])
            body_type = st.selectbox("Кузов", ['Sedan', 'Hatchback', 'SUV', 'Wagon', 'Coupe'])
            fuel_type = st.selectbox("Топливо", ['Petrol', 'Diesel', 'Hybrid', 'Electric'])
            doors = st.selectbox("Дверей", [2, 3, 4, 5])
            color = st.selectbox("Цвет", ['White', 'Black', 'Silver', 'Grey', 'Blue', 'Red', 'Green', 'Brown'])

        input_df = pd.DataFrame([{
            'brand': brand, 'model': model_car, 'year': year, 'mileage': mileage,
            'engine_capacity': engine, 'transmission': transmission,
            'body_type': body_type, 'fuel_type': fuel_type, 'color': color, 'doors': doors
        }])
        st.write("✅ Ваши данные:", input_df)

    else:
        st.subheader("Загрузка CSV-файла")
        uploaded = st.file_uploader("Загрузите .csv файл с признаками:", type=["csv"])
        if uploaded:
            input_df = pd.read_csv(uploaded)
            expected = ['brand', 'model', 'year', 'mileage', 'engine_capacity', 'transmission',
                        'body_type', 'fuel_type', 'color', 'doors']
            missing = [c for c in expected if c not in input_df.columns]
            if missing:
                st.error(f"Ошибка: отсутствуют колонки {missing}")
                st.stop()
            else:
                st.success(f"CSV загружен: {input_df.shape[0]} строк, {input_df.shape[1]} колонок")
                st.dataframe(input_df.head(), use_container_width=True)
        else:
            st.info("Загрузите CSV для предсказания")
            st.stop()

    # ===== ПРЕДСКАЗАНИЕ РЕГРЕССИИ =====
    if st.button("💰 Предсказать цену", type="primary"):
        try:
            if model_name.endswith("CatBoost"):
                pred = model.predict(input_df)
            elif model_name.endswith("TensorFlow)"):
                st.info("Используется нейросеть для предсказания...")
                prep = load_preprocessor("models_regression/preprocessor.pkl")
                X_proc = prep.transform(input_df)
                pred = model.predict(X_proc).flatten()
            else:
                pred = model.predict(input_df)
            
            for i, p in enumerate(pred):
                st.balloons()
                st.metric(
                    label=f"Автомобиль #{i+1}",
                    value=f"${p:,.2f} USD"
                )
                if p < 5000:
                    st.warning("⚠️ Низкая цена — возможно, автомобиль с высоким пробегом или старый.")
                elif p > 60000:
                    st.warning("⚠️ Высокая цена — премиальный сегмент. Проверьте характеристики.")
                else:
                    st.success("✅ Корректный диапазон цены для данной конфигурации.")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {str(e)}")

else:
    # ===== КЛАССИФИКАЦИЯ: ВВОД =====
    if input_method == "📝 Ручной ввод":
        st.subheader("Введите данные пациента")
        col1, col2, col3 = st.columns(3)
        with col1:
            high_bp = st.selectbox("Высокое давление?", ["Нет", "Да"])
            high_chol = st.selectbox("Высокий холестерин?", ["Нет", "Да"])
            chol_check = st.selectbox("Проверка холестерина за 5 лет?", ["Нет", "Да"])
            bmi = st.slider("BMI", 10.0, 60.0, 25.0, 0.5)
            smoker = st.selectbox("Курите?", ["Нет", "Да"])
            stroke = st.selectbox("Инсульт в анамнезе?", ["Нет", "Да"])
        with col2:
            heart = st.selectbox("Болезни сердца?", ["Нет", "Да"])
            phys = st.selectbox("Физическая активность?", ["Нет", "Да"])
            fruits = st.selectbox("Употребляете фрукты ежедневно?", ["Нет", "Да"])
            veggies = st.selectbox("Употребляете овощи ежедневно?", ["Нет", "Да"])
            alcohol = st.selectbox("Пьёте алкоголь сверхмерно?", ["Нет", "Да"])
            healthcare = st.selectbox("Есть ли медицинская страховка?", ["Нет", "Да"])
        with col3:
            no_doc = st.selectbox("Не могли обратиться к врачу по стоимости?", ["Нет", "Да"])
            gen_hlth = st.slider("Общее здоровье (1=отличное, 5=плохое)", 1, 5, 3)
            ment_hlth = st.slider("Кол-во дней плохого мент. здоровья в мес.", 0, 30, 0)
            phys_hlth = st.slider("Кол-во дней плохого физ. здоровья в мес.", 0, 30, 0)
            diff_walk = st.selectbox("Трудности с ходьбой?", ["Нет", "Да"])
            sex = st.selectbox("Пол", ["Женский", "Мужской"])
            age = st.slider("Возрастная группа (1=18-24 ..., 13=80+)", 1, 13, 5)
            education = st.slider("Образование (1=нет,..., 6=высшее)", 1, 6, 4)
            income = st.slider("Доход (1=<10K,..., 8=>75K)", 1, 8, 4)

        def yes_no(val):
            return 1 if val == "Да" else 0

        input_df = pd.DataFrame([{
            'HighBP': yes_no(high_bp), 'HighChol': yes_no(high_chol),
            'CholCheck': yes_no(chol_check), 'BMI': bmi,
            'Smoker': yes_no(smoker), 'Stroke': yes_no(stroke),
            'HeartDiseaseorAttack': yes_no(heart), 'PhysActivity': yes_no(phys),
            'Fruits': yes_no(fruits), 'Veggies': yes_no(veggies),
            'HvyAlcoholConsump': yes_no(alcohol), 'AnyHealthcare': yes_no(healthcare),
            'NoDocbcCost': yes_no(no_doc), 'GenHlth': gen_hlth,
            'MentHlth': ment_hlth, 'PhysHlth': phys_hlth,
            'DiffWalk': yes_no(diff_walk), 'Sex': yes_no(sex),
            'Age': age, 'Education': education, 'Income': income
        }])
        st.write("✅ Данные пациента:", input_df)

    else:
        st.subheader("Загрузка CSV-файла")
        uploaded = st.file_uploader("Загрузите .csv файл:", type=["csv"])
        if uploaded:
            input_df = pd.read_csv(uploaded)
            expected = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
            missing = [c for c in expected if c not in input_df.columns]
            if missing:
                st.error(f"Ошибка: отсутствуют колонки {missing}")
                st.stop()
            else:
                st.success(f"CSV загружен: {input_df.shape[0]} строк")
                st.dataframe(input_df.head(), use_container_width=True)
        else:
            st.info("Загрузите CSV для анализа")
            st.stop()

    # ===== ПРЕДСКАЗАНИЕ КЛАССИФИКАЦИИ =====
    if st.button("🩺 Диагностировать", type="primary"):
        try:
            if model_name.endswith("CatBoost"):
                pred = model.predict(input_df)
            elif model_name.endswith("TensorFlow)"):
                st.info("Используется нейросеть для диагностики...")
                scaler = load_scaler("models_classification/scaler.pkl")
                X_s = scaler.transform(input_df)
                probs = model.predict(X_s)
                pred = np.argmax(probs, axis=1)
            else:
                scaler = load_scaler("models_classification/scaler.pkl")
                X_s = scaler.transform(input_df)
                pred = model.predict(X_s)
            
            labels = {0: "🟢 Нет диабета", 1: "🟡 Преддиабет", 2: "🔴 Диабет"}
            for i, p in enumerate(pred):
                label = labels.get(int(p), "Неизвестно")
                if int(p) == 0:
                    st.success(f"Результат пациента #{i+1}: **{label}**")
                    st.info("Рекомендация: продолжайте здоровый образ жизни, регулярно проходите обследования.")
                elif int(p) == 1:
                    st.warning(f"Результат пациента #{i+1}: **{label}**")
                    st.info("Рекомендация: измените режим питания, увеличьте физическую активность, консультация у эндокринолога.")
                else:
                    st.error(f"Результат пациента #{i+1}: **{label}**")
                    st.info("Рекомендация: срочная консультация у эндокринолога, начало мониторинга глюкозы, возможно назначение терапии.")
        except Exception as e:
            st.error(f"Ошибка при диагностике: {str(e)}")
