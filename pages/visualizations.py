import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("📈 Визуализации данных")

task = st.selectbox("Выберите задачу:", [
    "Регрессия: Предсказание цены авто (cars)",
    "Классификация: Диагностика диабета (diabetes)"
])

is_regression = "Регрессия" in task

if is_regression and os.path.exists("data/cars.csv"):
    df = pd.read_csv("data/cars.csv")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Распределение цены", 
        "📉 Корреляция", 
        "🏭 По брендам", 
        "⛽ По топливу"
    ])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['price_usd'], kde=True, bins=40, color='steelblue', ax=ax)
        ax.set_title('Распределение цены автомобиля (USD)')
        ax.set_xlabel('Цена, USD')
        ax.set_ylabel('Количество')
        st.pyplot(fig)
        st.markdown("Гистограмма показывает, что распределение цены имеет **правый хвост** — большинство автомобилей стоит до 30 000 USD, а премиальные модели тянут хвост вправо.")
    
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_cols = ['year', 'mileage', 'engine_capacity', 'doors', 'price_usd']
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap='RdBu_r', center=0, ax=ax, fmt='.2f')
        ax.set_title('Корреляционная матрица числовых признаков')
        st.pyplot(fig)
        st.markdown("Тепловая карта корреляций: сильная **положительная** связь между `year` и `price_usd` и **отрицательная** между `mileage` и `price_usd`.")
    
    with tab3:
        fig, ax = plt.subplots(figsize=(12, 6))
        brand_price = df.groupby('brand')['price_usd'].mean().sort_values(ascending=False)
        sns.barplot(x=brand_price.index, y=brand_price.values, palette='viridis', ax=ax)
        ax.set_title('Средняя цена автомобиля по маркам')
        ax.set_xlabel('Марка')
        ax.set_ylabel('Средняя цена, USD')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown("Столбчатая диаграмма: **Mercedes, BMW и Audi** лидируют по средней цене, что соответствует реальному рынку.")
    
    with tab4:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='fuel_type', y='price_usd', data=df, palette='Set2', ax=ax)
        ax.set_title('Распределение цены в зависимости от типа топлива')
        ax.set_xlabel('Тип топлива')
        ax.set_ylabel('Цена, USD')
        st.pyplot(fig)
        st.markdown("Диаграмма размаха (boxplot): **Электрокары и гибриды** имеют более высокую цену и меньший разброс, чем автомобили на бензине.")

elif not is_regression and os.path.exists("data/diabetes.csv"):
    df = pd.read_csv("data/diabetes.csv")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🍰 Распределение классов", 
        "📉 Корреляция", 
        "⚖️ BMI по классам", 
        "💓 Здоровье"
    ])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(8, 6))
        counts = df['Diabetes_012'].value_counts().sort_index()
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        ax.pie(counts, labels=['Нет диабета (0)', 'Преддиабет (1)', 'Диабет (2)'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Распределение классов Diabetes_012')
        st.pyplot(fig)
        st.markdown("Круговая диаграмма: классы **несбалансированы** — класс 0 (нет диабета) составляет около половины выборки.")
    
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_features = ['BMI', 'GenHlth', 'Age', 'HighBP', 'HighChol', 'HeartDiseaseorAttack', 'Diabetes_012']
        sns.heatmap(df[corr_features].corr(), annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
        ax.set_title('Корреляционная матрица признаков и диабета')
        st.pyplot(fig)
        st.markdown("Тепловая карта: **GenHlth, BMI, Age и HighBP** наиболее коррелируют с наличием диабета.")
    
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x='Diabetes_012', y='BMI', data=df, palette='muted', ax=ax)
        ax.set_title('Распределение BMI по классам диабета')
        ax.set_xlabel('Класс Diabetes_012')
        ax.set_ylabel('BMI')
        ax.set_xticklabels(['Нет диабета', 'Преддиабет', 'Диабет'])
        st.pyplot(fig)
        st.markdown("Скрипичная диаграмма (violin plot): у больных диабетом BMI заметно **смещён вправо** — более высокие значения.")
    
    with tab4:
        fig, ax = plt.subplots(figsize=(10, 6))
        health_diabetes = pd.crosstab(df['GenHlth'], df['Diabetes_012'], normalize='index') * 100
        health_diabetes.plot(kind='bar', stacked=True, color=['#2ecc71', '#f1c40f', '#e74c3c'], ax=ax)
        ax.set_title('Процентное распределение диабета по общему здоровью')
        ax.set_xlabel('GenHlth (1=отличное, 5=плохое)')
        ax.set_ylabel('Процент, %')
        ax.legend(['Нет диабета', 'Преддиабет', 'Диабет'])
        st.pyplot(fig)
        st.markdown("Столбчатая диаграмма (stacked): при **плохом общем здоровье (GenHlth=5)** доля диабета резко возрастает.")

else:
    st.warning("Датасет не найден. Сначала запустите `generate_data.py`.")
