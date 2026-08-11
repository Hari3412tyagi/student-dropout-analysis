import pandas as pd

# 1. Load dataset
df = pd.read_csv("data/student data.csv")

print("===== DATASET OVERVIEW =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. First 5 records
print("\n===== FIRST 5 RECORDS =====")
print(df.head())

# 3. Column information
print("\n===== DATA TYPES =====")
print(df.dtypes)

# 4. Missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 5. Duplicate records
print("\n===== DUPLICATES =====")
print("Duplicate rows:", df.duplicated().sum())

# 6. Dropout distribution
print("\n===== DROPOUT DISTRIBUTION =====")
print(df["Dropout_Status"].value_counts())

# 7. Dropout percentage
print("\n===== DROPOUT PERCENTAGE =====")
print(df["Dropout_Status"].value_counts(normalize=True) * 100)
# ==========================================
# DROPOUT ANALYSIS
# ==========================================
# ==========================================
# DROPOUT ANALYSIS
# ==========================================

# Convert Yes/No into 1/0
df["Dropout_Flag"] = df["Dropout_Status"].map({
    "No": 0,
    "Yes": 1
})

print("\n===== SCHOOL-WISE DROPOUT =====")
school_dropout = df.groupby("School_ID")["Dropout_Flag"].mean() * 100
print(school_dropout.sort_values(ascending=False).head(10))


print("\n===== AREA-WISE DROPOUT =====")
area_dropout = df.groupby("Area")["Dropout_Flag"].mean() * 100
print(area_dropout.sort_values(ascending=False))


print("\n===== GENDER-WISE DROPOUT =====")
gender_dropout = df.groupby("Gender")["Dropout_Flag"].mean() * 100
print(gender_dropout.sort_values(ascending=False))


print("\n===== CASTE-WISE DROPOUT =====")
caste_dropout = df.groupby("Caste")["Dropout_Flag"].mean() * 100
print(caste_dropout.sort_values(ascending=False))


print("\n===== AGE-WISE DROPOUT =====")
age_dropout = df.groupby("Age")["Dropout_Flag"].mean() * 100
print(age_dropout.sort_values(ascending=False))


print("\n===== STANDARD-WISE DROPOUT =====")
standard_dropout = df.groupby("Standard")["Dropout_Flag"].mean() * 100
print(standard_dropout.sort_values(ascending=False))
# ==========================================
# FACTOR ANALYSIS
# ==========================================

print("\n===== INCOME-WISE DROPOUT =====")

df["Income_Group"] = pd.cut(
    df["Annual_Family_Income_INR"],
    bins=[0, 100000, 200000, 400000, 1000000],
    labels=[
        "Below 1 Lakh",
        "1-2 Lakh",
        "2-4 Lakh",
        "Above 4 Lakh"
    ]
)

income_dropout = df.groupby(
    "Income_Group",
    observed=True
)["Dropout_Flag"].mean() * 100

print(income_dropout)


print("\n===== ATTENDANCE-WISE DROPOUT =====")

df["Attendance_Group"] = pd.cut(
    df["Attendance_Percent"],
    bins=[0, 60, 75, 90, 100],
    labels=[
        "Below 60%",
        "60-75%",
        "75-90%",
        "90-100%"
    ]
)

attendance_dropout = df.groupby(
    "Attendance_Group",
    observed=True
)["Dropout_Flag"].mean() * 100

print(attendance_dropout)


print("\n===== DISTANCE-WISE DROPOUT =====")

df["Distance_Group"] = pd.cut(
    df["Distance_From_School_KM"],
    bins=[0, 2, 5, 10, 100],
    labels=[
        "0-2 KM",
        "2-5 KM",
        "5-10 KM",
        "Above 10 KM"
    ]
)

distance_dropout = df.groupby(
    "Distance_Group",
    observed=True
)["Dropout_Flag"].mean() * 100

print(distance_dropout)
# ==========================================
# CORRELATION ANALYSIS
# ==========================================

print("\n===== CORRELATION WITH DROPOUT =====")

correlation_data = df[
    [
        "Age",
        "Standard",
        "Annual_Family_Income_INR",
        "Attendance_Percent",
        "Distance_From_School_KM",
        "Dropout_Flag"
    ]
]

correlation = correlation_data.corr(numeric_only=True)

print(correlation["Dropout_Flag"].sort_values(ascending=False))
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Analysis of Student Dropout Factors")
plt.tight_layout()
plt.show()
# ==========================================
# MACHINE LEARNING MODEL
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("\n===== MACHINE LEARNING =====")

# Features
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

# Target
y = df["Dropout_Flag"]

# Categorical and numerical columns
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

# Preprocessing
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Logistic Regression pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\n===== MODEL PERFORMANCE =====")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))
# ==========================================
# RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier

print("\n===== RANDOM FOREST =====")

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

# Train
rf_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)

# Evaluation
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)

print("\n===== RANDOM FOREST CLASSIFICATION REPORT =====")
print(classification_report(y_test, rf_pred))

print("\n===== RANDOM FOREST CONFUSION MATRIX =====")
print(confusion_matrix(y_test, rf_pred))
print("\n===== MODEL COMPARISON =====")

print("\nLogistic Regression")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nRandom Forest")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)
# ==========================================
# STUDENT RISK PREDICTION
# ==========================================

print("\n===== STUDENT RISK PREDICTION =====")

# Example student
new_student = pd.DataFrame({
    "Area": ["Rural"],
    "Gender": ["Female"],
    "Caste": ["ST"],
    "Age": [15],
    "Standard": [10],
    "Annual_Family_Income_INR": [80000],
    "Attendance_Percent": [55],
    "Distance_From_School_KM": [12]
})

# Predict probability using Random Forest
risk_probability = rf_model.predict_proba(new_student)[0][1]

# Convert probability to percentage
risk_percentage = risk_probability * 100

# Risk level
if risk_percentage < 30:
    risk_level = "LOW"
elif risk_percentage < 60:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

print("\n===== PREDICTION RESULT =====")
print("Dropout Probability:", round(risk_percentage, 2), "%")
print("Risk Level:", risk_level)
# ==========================================
# RISK EXPLANATION & INTERVENTION
# ==========================================

print("\n===== RISK EXPLANATION =====")

student = new_student.iloc[0]

risk_factors = []

if student["Attendance_Percent"] < 60:
    risk_factors.append("Low attendance")

if student["Annual_Family_Income_INR"] < 100000:
    risk_factors.append("Low family income")

if student["Distance_From_School_KM"] > 10:
    risk_factors.append("Long distance from school")

if student["Standard"] >= 9:
    risk_factors.append("Higher standard")

if student["Area"] == "Rural":
    risk_factors.append("Rural area")

print("Risk Level:", risk_level)

print("\nPotential Risk Indicators:")

for factor in risk_factors:
    print("-", factor)


print("\n===== SUGGESTED INTERVENTION AREAS =====")

if student["Attendance_Percent"] < 60:
    print("- Attendance monitoring and follow-up")

if student["Annual_Family_Income_INR"] < 100000:
    print("- Review eligibility for financial/educational support")

if student["Distance_From_School_KM"] > 10:
    print("- Review transport and school accessibility support")

if student["Standard"] >= 9:
    print("- Academic support and retention monitoring")

if student["Area"] == "Rural":
    print("- Prioritize local school/community outreach")