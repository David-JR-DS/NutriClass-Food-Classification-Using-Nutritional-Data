import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="NutriClass",
    page_icon="🍎",
    layout="wide"
)

MODEL_PATH = Path("models/nutriclass_final_model.pkl")
DATA_PATH = Path("data/synthetic_food_dataset_imbalanced.csv")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Dataset file not found: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

st.title("🍎 NutriClass")
st.caption("Food classification and nutrition lookup using nutritional data")

tab1, tab2, tab3 = st.tabs([
    "⚡ Quick Prediction",
    "🧪 Full Model Input",
    "🥗 Food & Quantity Lookup"
])

# -----------------------------
# Helper
# -----------------------------
def predict_food_from_input(input_data: dict):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]
    classes = model.named_steps["classifier"].classes_

    prob_df = pd.DataFrame({
        "Food": classes,
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    return prediction, prob_df

# -----------------------------
# TAB 1: Quick Prediction
# -----------------------------
with tab1:
    st.subheader("Quick Prediction")
    st.write("Enter the most common nutritional details. Advanced model features can be adjusted in the Full Model Input tab.")

    c1, c2, c3 = st.columns(3)
    with c1:
        calories = st.number_input("Calories", min_value=0.0, value=300.0)
        protein = st.number_input("Protein (g)", min_value=0.0, value=15.0)
    with c2:
        fat = st.number_input("Fat (g)", min_value=0.0, value=12.0)
        carbs = st.number_input("Carbs (g)", min_value=0.0, value=30.0)
    with c3:
        sugar = st.number_input("Sugar (g)", min_value=0.0, value=6.0)
        serving_size = st.number_input("Serving Size (g)", min_value=1.0, value=150.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        meal_type = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner", "snack"])
    with c2:
        is_vegan = st.checkbox("Vegan")
    with c3:
        is_gluten_free = st.checkbox("Gluten Free", value=False)

    with st.expander("Advanced Details"):
        fiber = st.number_input("Fiber (g)", min_value=0.0, value=2.0)
        sodium = st.number_input("Sodium", min_value=0.0, value=300.0)
        cholesterol = st.number_input("Cholesterol", min_value=0.0, value=25.0)
        glycemic_index = st.number_input("Glycemic Index", min_value=0.0, value=60.0)
        water_content = st.number_input("Water Content", min_value=0.0, value=50.0)
        preparation_method = st.selectbox(
            "Preparation Method",
            ["baked", "fried", "grilled", "raw"]
        )

    if st.button("Predict Food", key="quick_predict", type="primary"):
        input_data = {
            "Calories": calories,
            "Protein": protein,
            "Fat": fat,
            "Carbs": carbs,
            "Sugar": sugar,
            "Fiber": fiber,
            "Sodium": sodium,
            "Cholesterol": cholesterol,
            "Glycemic_Index": glycemic_index,
            "Water_Content": water_content,
            "Serving_Size": serving_size,
            "Meal_Type": meal_type,
            "Preparation_Method": preparation_method,
            "Is_Vegan": is_vegan,
            "Is_Gluten_Free": is_gluten_free,
        }

        pred, probs = predict_food_from_input(input_data)

        st.success(f"Predicted Food: **{pred}**")
        st.metric("Model Confidence", f"{probs.iloc[0]['Probability'] * 100:.2f}%")
        st.dataframe(
            probs.head(3).assign(
                Probability=lambda x: (x["Probability"] * 100).round(2).astype(str) + "%"
            ),
            use_container_width=True,
            hide_index=True
        )

# -----------------------------
# TAB 2: Full Model Input
# -----------------------------
with tab2:
    st.subheader("Full Model Input")
    st.write("Evaluator mode: enter all 15 model features directly.")

    col1, col2, col3 = st.columns(3)

    with col1:
        calories_f = st.number_input("Calories", min_value=0.0, value=300.0, key="f_calories")
        protein_f = st.number_input("Protein (g)", min_value=0.0, value=15.0, key="f_protein")
        fat_f = st.number_input("Fat (g)", min_value=0.0, value=12.0, key="f_fat")
        carbs_f = st.number_input("Carbs (g)", min_value=0.0, value=30.0, key="f_carbs")
        sugar_f = st.number_input("Sugar (g)", min_value=0.0, value=6.0, key="f_sugar")

    with col2:
        fiber_f = st.number_input("Fiber (g)", min_value=0.0, value=2.0, key="f_fiber")
        sodium_f = st.number_input("Sodium", min_value=0.0, value=300.0, key="f_sodium")
        cholesterol_f = st.number_input("Cholesterol", min_value=0.0, value=25.0, key="f_cholesterol")
        gi_f = st.number_input("Glycemic Index", min_value=0.0, value=60.0, key="f_gi")
        water_f = st.number_input("Water Content", min_value=0.0, value=50.0, key="f_water")

    with col3:
        serving_f = st.number_input("Serving Size (g)", min_value=1.0, value=150.0, key="f_serving")
        meal_f = st.selectbox("Meal Type", ["breakfast", "lunch", "dinner", "snack"], key="f_meal")
        prep_f = st.selectbox("Preparation Method", ["baked", "fried", "grilled", "raw"], key="f_prep")
        vegan_f = st.checkbox("Vegan", key="f_vegan")
        gluten_f = st.checkbox("Gluten Free", value=False, key="f_gluten")

    if st.button("Predict with Full Model", key="full_predict", type="primary"):
        input_data = {
            "Calories": calories_f,
            "Protein": protein_f,
            "Fat": fat_f,
            "Carbs": carbs_f,
            "Sugar": sugar_f,
            "Fiber": fiber_f,
            "Sodium": sodium_f,
            "Cholesterol": cholesterol_f,
            "Glycemic_Index": gi_f,
            "Water_Content": water_f,
            "Serving_Size": serving_f,
            "Meal_Type": meal_f,
            "Preparation_Method": prep_f,
            "Is_Vegan": vegan_f,
            "Is_Gluten_Free": gluten_f,
        }

        pred, probs = predict_food_from_input(input_data)

        st.success(f"Predicted Food: **{pred}**")
        st.metric("Model Confidence", f"{probs.iloc[0]['Probability'] * 100:.2f}%")
        st.dataframe(
            probs.head(3).assign(
                Probability=lambda x: (x["Probability"] * 100).round(2).astype(str) + "%"
            ),
            use_container_width=True,
            hide_index=True
        )

# -----------------------------
# TAB 3: Food & Quantity Lookup
# -----------------------------
with tab3:
    st.subheader("Food & Quantity Lookup")
    st.write(
        "Select a food and quantity to view a representative nutritional profile "
        "from the dataset, scaled to the selected quantity."
    )

    food_list = sorted(df["Food_Name"].dropna().unique())
    selected_food = st.selectbox("Select Food", food_list)
    quantity = st.number_input("Quantity (g)", min_value=1.0, value=100.0, step=10.0)

    food_rows = df[df["Food_Name"] == selected_food].copy()

    numeric_food_cols = [
        "Calories", "Protein", "Fat", "Carbs", "Sugar", "Fiber",
        "Sodium", "Cholesterol", "Glycemic_Index", "Water_Content",
        "Serving_Size"
    ]

    # Use median profile for stability
    profile = food_rows[numeric_food_cols].median(numeric_only=True)

    # Scale numerical nutrition values by requested quantity relative to median serving size
    reference_serving = float(profile["Serving_Size"]) if profile["Serving_Size"] > 0 else 100.0
    scale = quantity / reference_serving

    scaled = profile.copy()
    for col in numeric_food_cols:
        if col != "Serving_Size":
            scaled[col] = scaled[col] * scale
    scaled["Serving_Size"] = quantity

    categorical_mode = {}
    for col in ["Meal_Type", "Preparation_Method", "Is_Vegan", "Is_Gluten_Free"]:
        mode_values = food_rows[col].mode()
        categorical_mode[col] = mode_values.iloc[0] if not mode_values.empty else None

    st.markdown("### Representative Nutritional Profile")

    display_profile = pd.DataFrame({
        "Feature": numeric_food_cols + [
            "Meal_Type", "Preparation_Method", "Is_Vegan", "Is_Gluten_Free"
        ],
        "Value": [
            round(float(scaled[col]), 2) for col in numeric_food_cols
        ] + [
            categorical_mode["Meal_Type"],
            categorical_mode["Preparation_Method"],
            categorical_mode["Is_Vegan"],
            categorical_mode["Is_Gluten_Free"]
        ]
    })

    st.dataframe(display_profile, use_container_width=True, hide_index=True)

    if st.button("Use This Profile for Classification", key="lookup_predict", type="primary"):
        lookup_input = {
            "Calories": scaled["Calories"],
            "Protein": scaled["Protein"],
            "Fat": scaled["Fat"],
            "Carbs": scaled["Carbs"],
            "Sugar": scaled["Sugar"],
            "Fiber": scaled["Fiber"],
            "Sodium": scaled["Sodium"],
            "Cholesterol": scaled["Cholesterol"],
            "Glycemic_Index": scaled["Glycemic_Index"],
            "Water_Content": scaled["Water_Content"],
            "Serving_Size": scaled["Serving_Size"],
            "Meal_Type": categorical_mode["Meal_Type"],
            "Preparation_Method": categorical_mode["Preparation_Method"],
            "Is_Vegan": categorical_mode["Is_Vegan"],
            "Is_Gluten_Free": categorical_mode["Is_Gluten_Free"],
        }

        pred, probs = predict_food_from_input(lookup_input)

        st.success(f"Model Prediction: **{pred}**")
        st.metric("Model Confidence", f"{probs.iloc[0]['Probability'] * 100:.2f}%")
        st.dataframe(
            probs.head(3).assign(
                Probability=lambda x: (x["Probability"] * 100).round(2).astype(str) + "%"
            ),
            use_container_width=True,
            hide_index=True
        )

st.markdown("---")
st.caption(
    "NutriClass | Final Gradient Boosting model | "
    "Test Accuracy 99.50% | Macro F1 99.52%"
)
