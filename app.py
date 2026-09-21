import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CGM Hypo Predictor MVP", layout="wide")
st.title("🩸 سیستم پیش‌بینی زودهنگام افت قند خون (CGM)")

# مدیریت بافر در Session State
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    val = st.number_input(
        "عدد قند خون (mg/dL):",
        min_value=30.0,
        max_value=450.0,
        value=115.0,
        step=1.0,
    )
with col2:
    if st.button("➕ ثبت خوانش (گذر ۵ دقیقه)"):
        st.session_state.history.append(val)
with col3:
    if st.button("🗑️ پاکسازی داده‌ها"):
        st.session_state.history = []

# اجرای الگوریتم روی داده‌های ثبت‌شده و نمایش جدول پیش‌بینی...
