import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Student Dropout Analytics",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data/student data.csv")

df["Dropout_Flag"] = df["Dropout_Status"].map({
    "No": 0,
    "Yes": 1
})

# ==========================================
# HEADER
# ==========================================

st.title("🎓 Student Dropout Analytics")
st.markdown(
    "### Early Risk Identification & Focused Intervention System"
)

st.divider()

# ==========================================
# KPI CARDS
# ==========================================

total_students = len(df)
total_dropouts = df["Dropout_Flag"].sum()
dropout_rate = (total_dropouts / total_students) * 100
non_dropouts = total_students - total_dropouts

col1, col2, col3, col4 = st.columns(4)

col1.metric("👨‍🎓 Total Students", total_students)
col2.metric("⚠️ Dropout Students", total_dropouts)
col3.metric("📊 Dropout Rate", f"{dropout_rate:.2f}%")
col4.metric("✅ Non-Dropout", non_dropouts)

st.divider()

# ==========================================
# SIDEBAR FILTER
# ==========================================

st.sidebar.header("🔎 Filters")

area_filter = st.sidebar.multiselect(
    "Select Area",
    options=df["Area"].unique(),
    default=df["Area"].unique()
)

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

caste_filter = st.sidebar.multiselect(
    "Select Caste",
    options=df["Caste"].unique(),
    default=df["Caste"].unique()
)

filtered_df = df[
    (df["Area"].isin(area_filter)) &
    (df["Gender"].isin(gender_filter)) &
    (df["Caste"].isin(caste_filter))
]

# ==========================================
# AREA-WISE ANALYSIS
# ==========================================

st.header("📍 Area-wise Dropout Analysis")

area_dropout = (
    filtered_df.groupby("Area")["Dropout_Flag"]
    .mean()
    .mul(100)
    .reset_index(name="Dropout Rate")
)

fig_area = px.bar(
    area_dropout,
    x="Area",
    y="Dropout Rate",
    title="Dropout Rate by Area",
    text_auto=".2f"
)

fig_area.update_layout(
    yaxis_title="Dropout Rate (%)",
    xaxis_title="Area"
)

st.plotly_chart(fig_area, use_container_width=True)

# ==========================================
# GENDER-WISE ANALYSIS
# ==========================================

st.header("👥 Gender-wise Dropout Analysis")

gender_dropout = (
    filtered_df.groupby("Gender")["Dropout_Flag"]
    .mean()
    .mul(100)
    .reset_index(name="Dropout Rate")
)

fig_gender = px.bar(
    gender_dropout,
    x="Gender",
    y="Dropout Rate",
    title="Dropout Rate by Gender",
    text_auto=".2f"
)

fig_gender.update_layout(
    yaxis_title="Dropout Rate (%)"
)

st.plotly_chart(fig_gender, use_container_width=True)

# ==========================================
# CASTE-WISE ANALYSIS
# ==========================================

st.header("🧑‍🤝‍🧑 Caste-wise Dropout Analysis")

caste_dropout = (
    filtered_df.groupby("Caste")["Dropout_Flag"]
    .mean()
    .mul(100)
    .reset_index(name="Dropout Rate")
)

fig_caste = px.bar(
    caste_dropout,
    x="Caste",
    y="Dropout Rate",
    title="Dropout Rate by Caste",
    text_auto=".2f"
)

fig_caste.update_layout(
    yaxis_title="Dropout Rate (%)"
)

st.plotly_chart(fig_caste, use_container_width=True)

# ==========================================
# STANDARD-WISE ANALYSIS
# ==========================================

st.header("🎓 Standard-wise Dropout Analysis")

standard_dropout = (
    filtered_df.groupby("Standard")["Dropout_Flag"]
    .mean()
    .mul(100)
    .reset_index(name="Dropout Rate")
)

fig_standard = px.bar(
    standard_dropout,
    x="Standard",
    y="Dropout Rate",
    title="Dropout Rate by Standard",
    text_auto=".2f"
)

fig_standard.update_layout(
    yaxis_title="Dropout Rate (%)",
    xaxis_title="Standard"
)

st.plotly_chart(fig_standard, use_container_width=True)

# ==========================================
# SCHOOL-WISE TOP 10
# ==========================================

st.header("🏫 Top 10 High-Dropout Schools")

school_dropout = (
    filtered_df.groupby("School_ID")["Dropout_Flag"]
    .mean()
    .mul(100)
    .reset_index(name="Dropout Rate")
    .sort_values("Dropout Rate", ascending=False)
    .head(10)
)

fig_school = px.bar(
    school_dropout,
    x="School_ID",
    y="Dropout Rate",
    title="Top 10 Schools by Dropout Rate",
    text_auto=".2f"
)

fig_school.update_layout(
    yaxis_title="Dropout Rate (%)"
)

st.plotly_chart(fig_school, use_container_width=True)

# ==========================================
# FINAL SUMMARY
# ==========================================

st.divider()

st.success(
    "🎯 This dashboard identifies high-dropout groups and "
    "supports focused intervention planning."
)
# ==========================================
# ML RISK PREDICTION
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

st.divider()

st.header("⚠️ Student Dropout Risk Prediction")
st.write(
    "Enter student information to estimate dropout risk "
    "and identify possible intervention areas."
)

# ------------------------------------------
# Prepare ML data
# ------------------------------------------

X = df[
    [
        "Area",
        "Gender",
        "Caste",
        "Age",
        "Standard",
        "Annual_Family_Income_INR",
        "Attendance_Percent",
        "Distance_From_School_KM"
    ]
]

y = df["Dropout_Flag"]

categorical_features = [
    "Area",
    "Gender",
    "Caste"
]

numerical_features = [
    "Age",
    "Standard",
    "Annual_Family_Income_INR",
    "Attendance_Percent",
    "Distance_From_School_KM"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

rf_model.fit(X_train, y_train)

# ------------------------------------------
# Student Input
# ------------------------------------------

st.subheader("Enter Student Details")

col1, col2, col3 = st.columns(3)

with col1:
    area = st.selectbox(
        "Area",
        df["Area"].unique()
    )

    gender = st.selectbox(
        "Gender",
        df["Gender"].unique()
    )

    caste = st.selectbox(
        "Caste",
        df["Caste"].unique()
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=int(df["Age"].min()),
        max_value=int(df["Age"].max()),
        value=15
    )

    standard = st.number_input(
        "Standard",
        min_value=int(df["Standard"].min()),
        max_value=int(df["Standard"].max()),
        value=10
    )

    income = st.number_input(
        "Annual Family Income (₹)",
        min_value=0,
        value=100000,
        step=10000
    )

with col3:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=75
    )

    distance = st.number_input(
        "Distance From School (KM)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.5
    )

# ------------------------------------------
# Prediction
# ------------------------------------------

if st.button("🔍 Predict Dropout Risk"):

    new_student = pd.DataFrame({
        "Area": [area],
        "Gender": [gender],
        "Caste": [caste],
        "Age": [age],
        "Standard": [standard],
        "Annual_Family_Income_INR": [income],
        "Attendance_Percent": [attendance],
        "Distance_From_School_KM": [distance]
    })

    probability = rf_model.predict_proba(
        new_student
    )[0][1]

    risk_percentage = probability * 100

    if risk_percentage < 30:
        risk_level = "LOW"
    elif risk_percentage < 60:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    st.divider()

    st.subheader("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Dropout Probability",
        f"{risk_percentage:.2f}%"
    )

    result_col2.metric(
        "Risk Level",
        risk_level
    )

    # --------------------------------------
    # Risk Indicators
    # --------------------------------------

    risk_factors = []

    if attendance < 60:
        risk_factors.append("Low attendance")

    if income < 100000:
        risk_factors.append("Low family income")

    if distance > 10:
        risk_factors.append("Long distance from school")

    if standard >= 9:
        risk_factors.append("Higher standard")

    if area == "Rural":
        risk_factors.append("Rural area")

    st.subheader("⚠️ Potential Risk Indicators")

    if risk_factors:
        for factor in risk_factors:
            st.write("🔸", factor)
    else:
        st.success("No major rule-based risk indicators detected.")

    # --------------------------------------
    # Intervention Areas
    # --------------------------------------

    st.subheader("🎯 Suggested Intervention Areas")

    interventions = []

    if attendance < 60:
        interventions.append(
            "Attendance monitoring and follow-up"
        )

    if income < 100000:
        interventions.append(
            "Review eligibility for financial/educational support"
        )

    if distance > 10:
        interventions.append(
            "Review transport and school accessibility support"
        )

    if standard >= 9:
        interventions.append(
            "Academic support and retention monitoring"
        )

    if area == "Rural":
        interventions.append(
            "Local school/community outreach"
        )

    if interventions:
        for intervention in interventions:
            st.write("🎯", intervention)
    else:
        st.info(
            "Continue regular student monitoring and support."
        )

    st.caption(
        "Note: This system provides a risk estimate and "
        "decision-support information. Final intervention "
        "decisions should be made by authorized school or "
        "education officials."
    )
    # ==========================================
# INTERVENTION PRIORITY
# ==========================================

st.divider()

st.header("🎯 Intervention Priority Ranking")

st.write(
    "Schools are prioritized according to their observed dropout rate "
    "to support focused intervention planning."
)

priority_df = (
    filtered_df.groupby("School_ID")["Dropout_Flag"]
    .mean()
    .mul(100)
    .reset_index(name="Dropout Rate")
)

def assign_priority(rate):

    if rate >= 60:
        return "🔴 VERY HIGH"

    elif rate >= 45:
        return "🟠 HIGH"

    elif rate >= 30:
        return "🟡 MODERATE"

    else:
        return "🟢 LOW"


priority_df["Priority"] = priority_df["Dropout Rate"].apply(
    assign_priority
)

priority_df = priority_df.sort_values(
    "Dropout Rate",
    ascending=False
)

st.dataframe(
    priority_df.head(20),
    use_container_width=True,
    hide_index=True
)