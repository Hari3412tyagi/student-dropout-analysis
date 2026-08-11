import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Dropout Analytics",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("data/student data.csv")

    df["Dropout_Flag"] = df["Dropout_Status"].map({
        "No": 0,
        "Yes": 1
    })

    return df


df = load_data()


# =========================================================
# TRAIN ML MODEL
# =========================================================

@st.cache_resource
def train_model(df):

    features = [
        "Area",
        "Gender",
        "Caste",
        "Age",
        "Standard",
        "Annual_Family_Income_INR",
        "Attendance_Percent",
        "Distance_From_School_KM"
    ]

    X = df[features]
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

    model = Pipeline(
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

    model.fit(X_train, y_train)

    return model


rf_model = train_model(df)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🎓 Student Dropout System")

st.sidebar.markdown(
    "### Early Risk Identification"
)

page = st.sidebar.radio(
    "Navigation",
    [
    "📊 Overview",
    "📈 Dropout Analysis",
    "🤖 Risk Prediction",
    "🎯 Intervention Priority",
    "🧠 Model Performance"
]
)

st.sidebar.divider()

st.sidebar.info(
    "Decision-support prototype for identifying "
    "dropout patterns and potential high-risk cases."
)


# =========================================================
# COMMON FILTERS
# =========================================================

if page == "📈 Dropout Analysis":

    st.sidebar.header("🔎 Analysis Filters")

    area_filter = st.sidebar.multiselect(
        "Area",
        options=df["Area"].unique(),
        default=df["Area"].unique()
    )

    gender_filter = st.sidebar.multiselect(
        "Gender",
        options=df["Gender"].unique(),
        default=df["Gender"].unique()
    )

    caste_filter = st.sidebar.multiselect(
        "Caste",
        options=df["Caste"].unique(),
        default=df["Caste"].unique()
    )

    filtered_df = df[
        (df["Area"].isin(area_filter)) &
        (df["Gender"].isin(gender_filter)) &
        (df["Caste"].isin(caste_filter))
    ]

else:

    filtered_df = df


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "📊 Overview":

    st.title("🎓 Student Dropout Analytics")

    st.markdown(
        "### Early Risk Identification & Focused Intervention System"
    )

    st.divider()

    total_students = len(df)

    total_dropouts = int(
        df["Dropout_Flag"].sum()
    )

    non_dropouts = total_students - total_dropouts

    dropout_rate = (
        total_dropouts / total_students
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👨‍🎓 Total Students",
        total_students
    )

    col2.metric(
        "⚠️ Dropout Students",
        total_dropouts
    )

    col3.metric(
        "📊 Dropout Rate",
        f"{dropout_rate:.2f}%"
    )

    col4.metric(
        "✅ Non-Dropout",
        non_dropouts
    )

    st.divider()

    st.subheader("📌 Project Objective")

    st.write(
        "The system analyzes student dropout patterns across "
        "school, area, gender, caste, age and standard, "
        "and supports early risk identification and focused "
        "intervention planning."
    )

    st.subheader("🔄 System Workflow")

    st.markdown(
        """
        **Student Data**
        ↓  
        **Data Analysis**
        ↓  
        **Dropout Pattern Identification**
        ↓  
        **ML Risk Prediction**
        ↓  
        **Risk Identification**
        ↓  
        **Intervention Priority**
        """
    )

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 2 — DROPOUT ANALYSIS
# =========================================================

elif page == "📈 Dropout Analysis":

    st.title("📈 Dropout Analysis")

    st.write(
        "Analyze dropout patterns across different "
        "student and school-level categories."
    )

    st.divider()

    # ---------------- AREA ----------------

    st.subheader("📍 Area-wise Dropout")

    area_dropout = (
        filtered_df
        .groupby("Area")["Dropout_Flag"]
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
        yaxis_title="Dropout Rate (%)"
    )

    st.plotly_chart(
        fig_area,
        use_container_width=True
    )

    # ---------------- GENDER ----------------

    st.subheader("👥 Gender-wise Dropout")

    gender_dropout = (
        filtered_df
        .groupby("Gender")["Dropout_Flag"]
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

    st.plotly_chart(
        fig_gender,
        use_container_width=True
    )

    # ---------------- CASTE ----------------

    st.subheader("🧑‍🤝‍🧑 Caste-wise Dropout")

    caste_dropout = (
        filtered_df
        .groupby("Caste")["Dropout_Flag"]
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

    st.plotly_chart(
        fig_caste,
        use_container_width=True
    )

    # ---------------- STANDARD ----------------

    st.subheader("🎓 Standard-wise Dropout")

    standard_dropout = (
        filtered_df
        .groupby("Standard")["Dropout_Flag"]
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

    st.plotly_chart(
        fig_standard,
        use_container_width=True
    )

    # ---------------- SCHOOL ----------------

    st.subheader("🏫 Top 10 High-Dropout Schools")

    school_dropout = (
        filtered_df
        .groupby("School_ID")["Dropout_Flag"]
        .mean()
        .mul(100)
        .reset_index(name="Dropout Rate")
        .sort_values(
            "Dropout Rate",
            ascending=False
        )
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

    st.plotly_chart(
        fig_school,
        use_container_width=True
    )


# =========================================================
# PAGE 3 — RISK PREDICTION
# =========================================================

elif page == "🤖 Risk Prediction":

    st.title("🤖 Student Dropout Risk Prediction")

    st.write(
        "Enter student information to estimate dropout "
        "risk and identify potential risk indicators."
    )

    st.divider()

    st.subheader("📝 Student Information")

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

    st.divider()

    if st.button(
        "🔍 Predict Dropout Risk",
        use_container_width=True
    ):

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
            risk_message = "Low priority monitoring"

        elif risk_percentage < 60:

            risk_level = "MEDIUM"
            risk_message = "Needs attention and monitoring"

        else:

            risk_level = "HIGH"
            risk_message = "Priority review recommended"

        st.subheader("📊 Prediction Result")

        result1, result2 = st.columns(2)

        result1.metric(
            "Dropout Probability",
            f"{risk_percentage:.2f}%"
        )

        result2.metric(
            "Risk Level",
            risk_level
        )

        st.info(risk_message)

        # -----------------------------------------
        # RISK INDICATORS
        # -----------------------------------------

        st.subheader("⚠️ Potential Risk Indicators")

        risk_factors = []

        if attendance < 60:
            risk_factors.append(
                "Low attendance"
            )

        if income < 100000:
            risk_factors.append(
                "Low family income"
            )

        if distance > 10:
            risk_factors.append(
                "Long distance from school"
            )

        if standard >= 9:
            risk_factors.append(
                "Higher standard"
            )

        if area == "Rural":
            risk_factors.append(
                "Rural area"
            )

        if risk_factors:

            for factor in risk_factors:

                st.write(
                    "🔸",
                    factor
                )

        else:

            st.success(
                "No major rule-based risk indicators detected."
            )

        # -----------------------------------------
        # INTERVENTION
        # -----------------------------------------

        st.subheader(
            "🎯 Suggested Intervention Areas"
        )

        interventions = []

        if attendance < 60:

            interventions.append(
                "Attendance monitoring and follow-up"
            )

        if income < 100000:

            interventions.append(
                "Review eligibility for "
                "financial/educational support"
            )

        if distance > 10:

            interventions.append(
                "Review transport and "
                "school accessibility support"
            )

        if standard >= 9:

            interventions.append(
                "Academic support and "
                "retention monitoring"
            )

        if area == "Rural":

            interventions.append(
                "Local school/community outreach"
            )

        if interventions:

            for intervention in interventions:

                st.write(
                    "🎯",
                    intervention
                )

        else:

            st.info(
                "Continue regular student monitoring and support."
            )

        st.caption(
            "This system provides a risk estimate and "
            "decision-support information. Final intervention "
            "decisions should be made by authorized school "
            "or education officials."
        )


# =========================================================
# PAGE 4 — INTERVENTION PRIORITY
# =========================================================

elif page == "🎯 Intervention Priority":

    st.title("🎯 Intervention Priority")

    st.write(
        "Identify schools with higher observed dropout rates "
        "for focused intervention planning."
    )

    st.divider()

    priority_df = (
        df
        .groupby("School_ID")["Dropout_Flag"]
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

    priority_df["Priority"] = (
        priority_df["Dropout Rate"]
        .apply(assign_priority)
    )

    priority_df = priority_df.sort_values(
        "Dropout Rate",
        ascending=False
    )

    # -----------------------------------------
    # PRIORITY SUMMARY
    # -----------------------------------------

    very_high = len(
        priority_df[
            priority_df["Priority"] == "🔴 VERY HIGH"
        ]
    )

    high = len(
        priority_df[
            priority_df["Priority"] == "🟠 HIGH"
        ]
    )

    moderate = len(
        priority_df[
            priority_df["Priority"] == "🟡 MODERATE"
        ]
    )

    low = len(
        priority_df[
            priority_df["Priority"] == "🟢 LOW"
        ]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🔴 Very High",
        very_high
    )

    col2.metric(
        "🟠 High",
        high
    )

    col3.metric(
        "🟡 Moderate",
        moderate
    )

    col4.metric(
        "🟢 Low",
        low
    )

    st.divider()

    st.subheader(
        "🏫 School Priority Ranking"
    )

    st.dataframe(
        priority_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "🎯 Intervention Framework"
    )

    st.markdown(
        """
        🔴 **VERY HIGH**  
        Immediate review and priority intervention planning.

        🟠 **HIGH**  
        Detailed monitoring and targeted support planning.

        🟡 **MODERATE**  
        Regular monitoring with preventive support.

        🟢 **LOW**  
        Continue routine monitoring.
        """
    )

    st.warning(
        "Priority ranking is based on observed dropout rates "
        "in the current dataset. It should support—not replace—"
        "human decision-making."
    )
    # =========================================================
# PAGE 5 — MODEL PERFORMANCE
# =========================================================

elif page == "🧠 Model Performance":

    st.title("🧠 ML Model Performance")

    st.write(
        "Evaluation of the Random Forest model used for "
        "student dropout risk prediction."
    )

    st.divider()

    # Prepare test data
    features = [
        "Area",
        "Gender",
        "Caste",
        "Age",
        "Standard",
        "Annual_Family_Income_INR",
        "Attendance_Percent",
        "Distance_From_School_KM"
    ]

    X = df[features]
    y = df["Dropout_Flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Train a fresh model for evaluation
    evaluation_model = train_model(df)

    # Predictions
    y_pred = evaluation_model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # =============================================
    # METRIC CARDS
    # =============================================

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

    st.divider()

    # =============================================
    # CONFUSION MATRIX
    # =============================================

    st.subheader("🔲 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Dropout", "Actual: Dropout"],
        columns=["Predicted: No Dropout", "Predicted: Dropout"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # =============================================
    # EXPLANATION
    # =============================================

    st.subheader("📚 Metric Interpretation")

    st.markdown(
        """
        **Accuracy**  
        Overall percentage of correct predictions.

        **Precision**  
        Among students predicted as dropout-risk,
        how many were actually dropouts.

        **Recall**  
        Among actual dropout students,
        how many were correctly identified by the model.

        **F1 Score**  
        Balanced measure combining Precision and Recall.
        """
    )

    st.info(
        "For this project, Recall is especially important because "
        "missing an at-risk student may result in a missed "
        "opportunity for early intervention."
    )

    st.caption(
        "Model evaluation is based on an 80/20 train-test split "
        "with a fixed random state for reproducibility."
    )
    # ---------------- AGE ----------------

st.subheader("👤 Age-wise Dropout")

age_dropout = (
    filtered_df
    .groupby("Age")["Dropout_Flag"]
    .mean()
    .mul(100)
    .reset_index(name="Dropout Rate")
    .sort_values("Age")
)

fig_age = px.line(
    age_dropout,
    x="Age",
    y="Dropout Rate",
    markers=True,
    title="Dropout Rate by Age"
)

fig_age.update_layout(
    yaxis_title="Dropout Rate (%)",
    xaxis_title="Age"
)

st.plotly_chart(
    fig_age,
    use_container_width=True
)


# ---------------- INCOME ----------------

st.subheader("💰 Income-wise Dropout")

df_income = filtered_df.copy()

df_income["Income_Group"] = pd.cut(
    df_income["Annual_Family_Income_INR"],
    bins=[0, 100000, 200000, 400000, float("inf")],
    labels=[
        "Below 1 Lakh",
        "1-2 Lakh",
        "2-4 Lakh",
        "Above 4 Lakh"
    ],
    include_lowest=True
)

income_dropout = (
    df_income
    .groupby("Income_Group", observed=False)["Dropout_Flag"]
    .mean()
    .mul(100)
    .reset_index(name="Dropout Rate")
)

fig_income = px.bar(
    income_dropout,
    x="Income_Group",
    y="Dropout Rate",
    title="Dropout Rate by Family Income",
    text_auto=".2f"
)

fig_income.update_layout(
    yaxis_title="Dropout Rate (%)",
    xaxis_title="Family Income"
)

st.plotly_chart(
    fig_income,
    use_container_width=True
)


# ---------------- ATTENDANCE ----------------

st.subheader("📅 Attendance-wise Dropout")

df_attendance = filtered_df.copy()

df_attendance["Attendance_Group"] = pd.cut(
    df_attendance["Attendance_Percent"],
    bins=[0, 60, 75, 90, 100],
    labels=[
        "Below 60%",
        "60-75%",
        "75-90%",
        "90-100%"
    ],
    include_lowest=True
)

attendance_dropout = (
    df_attendance
    .groupby(
        "Attendance_Group",
        observed=False
    )["Dropout_Flag"]
    .mean()
    .mul(100)
    .reset_index(name="Dropout Rate")
)

fig_attendance = px.bar(
    attendance_dropout,
    x="Attendance_Group",
    y="Dropout Rate",
    title="Dropout Rate by Attendance",
    text_auto=".2f"
)

fig_attendance.update_layout(
    yaxis_title="Dropout Rate (%)",
    xaxis_title="Attendance"
)

st.plotly_chart(
    fig_attendance,
    use_container_width=True
)


# ---------------- DISTANCE ----------------

st.subheader("🚌 Distance-wise Dropout")

df_distance = filtered_df.copy()

df_distance["Distance_Group"] = pd.cut(
    df_distance["Distance_From_School_KM"],
    bins=[0, 2, 5, 10, float("inf")],
    labels=[
        "0-2 KM",
        "2-5 KM",
        "5-10 KM",
        "Above 10 KM"
    ],
    include_lowest=True
)

distance_dropout = (
    df_distance
    .groupby(
        "Distance_Group",
        observed=False
    )["Dropout_Flag"]
    .mean()
    .mul(100)
    .reset_index(name="Dropout Rate")
)

fig_distance = px.bar(
    distance_dropout,
    x="Distance_Group",
    y="Dropout Rate",
    title="Dropout Rate by Distance from School",
    text_auto=".2f"
)

fig_distance.update_layout(
    yaxis_title="Dropout Rate (%)",
    xaxis_title="Distance"
)

st.plotly_chart(
    fig_distance,
    use_container_width=True
)
# =========================================================
# KEY INSIGHTS
# =========================================================

st.divider()

st.header("💡 Key Insights & Early Warning Signals")

# Area insight
area_insight = (
    filtered_df.groupby("Area")["Dropout_Flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

# Attendance insight
attendance_insight = (
    df_attendance.groupby("Attendance_Group", observed=False)["Dropout_Flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

# Income insight
income_insight = (
    df_income.groupby("Income_Group", observed=False)["Dropout_Flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

# Distance insight
distance_insight = (
    df_distance.groupby("Distance_Group", observed=False)["Dropout_Flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

# Standard insight
standard_insight = (
    filtered_df.groupby("Standard")["Dropout_Flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

col1, col2 = st.columns(2)

with col1:

    if not area_insight.empty:

        highest_area = area_insight.index[0]
        highest_area_rate = area_insight.iloc[0]

        st.warning(
            f"📍 **Highest observed area dropout:** "
            f"{highest_area} ({highest_area_rate:.2f}%)"
        )

    if not attendance_insight.empty:

        highest_attendance_group = attendance_insight.index[0]
        highest_attendance_rate = attendance_insight.iloc[0]

        st.warning(
            f"📅 **Highest observed attendance-group dropout:** "
            f"{highest_attendance_group} "
            f"({highest_attendance_rate:.2f}%)"
        )

with col2:

    if not income_insight.empty:

        highest_income_group = income_insight.index[0]
        highest_income_rate = income_insight.iloc[0]

        st.info(
            f"💰 **Highest observed income-group dropout:** "
            f"{highest_income_group} "
            f"({highest_income_rate:.2f}%)"
        )

    if not distance_insight.empty:

        highest_distance_group = distance_insight.index[0]
        highest_distance_rate = distance_insight.iloc[0]

        st.warning(
            f"🚌 **Highest observed distance-group dropout:** "
            f"{highest_distance_group} "
            f"({highest_distance_rate:.2f}%)"
        )

if not standard_insight.empty:

    highest_standard = standard_insight.index[0]
    highest_standard_rate = standard_insight.iloc[0]

    st.info(
        f"🎓 **Highest observed standard dropout:** "
        f"Standard {highest_standard} "
        f"({highest_standard_rate:.2f}%)"
    )

st.caption(
    "These are observed patterns in the dataset and should "
    "not be interpreted as proof that a factor directly causes dropout."
)