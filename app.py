import streamlit as st

st.set_page_config(
    page_title="ML Inference Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 ML Inference Dashboard")
st.markdown("""
**Добро пожаловать!** Это веб-приложение (дашборд) для инференса (вывода) 
моделей машинного обучения и визуализации данных.

Используйте боковую панель для навигации по страницам:
- 🧑‍💻 **О разработчике** — информация об авторе и стеке технологий
- 📁 **О наборе данных** — описание датасетов и EDA
- 📈 **Визуализации** — 4+ вида графиков (Matplotlib, Seaborn)
- 🔮 **Предсказание** — загрузка данных и инференс 6 моделей ML
""")

st.sidebar.success("Выберите страницу выше ⬆️")
