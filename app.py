import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("Student_Performance.csv")

@st.cache_resource
def train_models():
    df = load_data().drop_duplicates().reset_index(drop=True)
    target = "final_grade"
    excluded = ["student_id", "math_score", "science_score", "english_score", "overall_score", target]
    X = df.drop(columns=excluded)
    y = df[target]
    score = df["overall_score"]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, score, test_size=0.2, random_state=42, stratify=y
    )

    grade_model = Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1
        ))
    ])
    score_model = Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, random_state=42, n_jobs=-1
        ))
    ])

    grade_model.fit(X_train, y_train)
    score_model.fit(X_train, s_train)

    pred = grade_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "mae": mean_absolute_error(s_test, np.clip(score_model.predict(X_test), 0, 100)),
        "r2": r2_score(s_test, np.clip(score_model.predict(X_test), 0, 100))
    }
    return df, X, grade_model, score_model, metrics

df, X, grade_model, score_model, metrics = train_models()

st.title("🎓 Student Academic Performance Predictor")
st.write("Predict a student's final grade and estimated overall score from study habits and background.")
st.caption("This is an educational machine-learning demo. Predictions are estimates, not official grades.")

tab1, tab2, tab3 = st.tabs(["Predict a Student", "Dataset Overview", "Model Performance"])

with tab1:
    st.subheader("Enter Student Details")
    left, right = st.columns(2)
    with left:
        age = st.slider("Age", 14, 19, 17)
        gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique()))
        school_type = st.selectbox("School type", sorted(df["school_type"].dropna().unique()))
        parent_education = st.selectbox("Parent education", sorted(df["parent_education"].dropna().unique()))
        study_method = st.selectbox("Study method", sorted(df["study_method"].dropna().unique()))
    with right:
        study_hours = st.slider("Study hours per day", 0.5, 8.0, 6.0, 0.5)
        attendance = st.slider("Attendance (%)", 50.0, 100.0, 90.0, 1.0)
        internet_access = st.selectbox("Internet access", sorted(df["internet_access"].dropna().unique()))
        travel_time = st.selectbox("Travel time to school", sorted(df["travel_time"].dropna().unique()))
        extra_activities = st.selectbox("Extra activities", sorted(df["extra_activities"].dropna().unique()))

    if st.button("Predict Performance", type="primary", use_container_width=True):
        student = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "school_type": school_type,
            "parent_education": parent_education,
            "study_hours": study_hours,
            "attendance_percentage": attendance,
            "internet_access": internet_access,
            "travel_time": travel_time,
            "extra_activities": extra_activities,
            "study_method": study_method
        }], columns=X.columns)

        grade = str(grade_model.predict(student)[0]).upper()
        probabilities = grade_model.predict_proba(student)[0]
        score = float(np.clip(score_model.predict(student)[0], 0, 100))

        c1, c2 = st.columns(2)
        c1.metric("Predicted Final Grade", grade)
        c2.metric("Estimated Overall Score", f"{score:.1f} / 100")

        prob_df = pd.DataFrame({
            "Grade": [str(g).upper() for g in grade_model.classes_],
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)
        st.subheader("Grade Probability")
        st.bar_chart(prob_df.set_index("Grade"))

with tab2:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Duplicate Rows", int(df.duplicated().sum()))
    st.write("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.write("Final grade distribution")
    grade_counts = df["final_grade"].value_counts().sort_index()
    st.bar_chart(grade_counts)
    st.write("Missing values by column")
    missing = df.isnull().sum().rename("Missing values").to_frame()
    st.dataframe(missing, use_container_width=True)

with tab3:
    st.subheader("Test-set Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grade Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Macro F1-score", f"{metrics['f1']:.3f}")
    c3.metric("Score MAE", f"{metrics['mae']:.2f}")
    c4.metric("Score R²", f"{metrics['r2']:.3f}")
    st.write("Accuracy and F1-score evaluate grade classification. MAE and R² evaluate score prediction.")
