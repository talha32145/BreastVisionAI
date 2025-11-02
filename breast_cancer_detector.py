from sklearn.svm import LinearSVC
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,precision_score,classification_report
import streamlit as st
import google.generativeai as genai
import seaborn as sns 

genai.configure(api_key="YOUR_API_KEY")
models=genai.GenerativeModel("gemini-2.5-pro")

df=pd.read_csv("Breast_cancer_dataset.csv")

df["diagnosis"]=pd.factorize(df["diagnosis"])[0]

x = df[['radius_mean','texture_mean','perimeter_mean','area_mean',
        'smoothness_mean','compactness_mean','concavity_mean',
        'concave points_mean','symmetry_mean','fractal_dimension_mean',
        'radius_se','texture_se','perimeter_se','area_se',
        'smoothness_se','compactness_se','concavity_se',
        'concave points_se','symmetry_se','fractal_dimension_se',
        'radius_worst','texture_worst','perimeter_worst','area_worst',
        'smoothness_worst','compactness_worst','concavity_worst',
        'concave points_worst','symmetry_worst','fractal_dimension_worst']]

y=df["diagnosis"]




x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearSVC(random_state=42,max_iter=1000,C=0.01,penalty="l2",class_weight="balanced")

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

st.sidebar.title("🧬 BreastVision AI")
st.sidebar.markdown("An AI-powered Breast Cancer Detection System")


st.sidebar.header("📌 Navigation")
st.sidebar.info(
    "Use the **tabs above** to:\n"
    "- 💬 Chat with the AI using 30 features\n"
    "- 📷 Upload and analyze mammogram\n"
    "- 📊 View model performance"
)

if st.sidebar.checkbox("👀 Preview Dataset"):
    st.sidebar.write(df.head())

st.sidebar.warning("⚠️ BreastVision AI is a research prototype. It cannot replace medical diagnosis.")

tab1,tab2,tab3=st.tabs(["Chat bot with 30 numeric values","x-ray anaylsis with BreastVision AI","Model Evalution Result"])

with tab1:
    st.set_page_config(page_title="Breast Cancer Detector", page_icon="🩺", layout="wide")

    st.title("🩺 Breast Cancer Detection Chatbot")
    st.caption("💬 Talk with the ML model to detect whether tumor is **Malignant** or **Benign**.")

    # List of 30 feature names
    feature_names = [
        'radius_mean','texture_mean','perimeter_mean','area_mean',
        'smoothness_mean','compactness_mean','concavity_mean',
        'concave points_mean','symmetry_mean','fractal_dimension_mean',
        'radius_se','texture_se','perimeter_se','area_se',
        'smoothness_se','compactness_se','concavity_se',
        'concave points_se','symmetry_se','fractal_dimension_se',
        'radius_worst','texture_worst','perimeter_worst','area_worst',
        'smoothness_worst','compactness_worst','concavity_worst',
        'concave points_worst','symmetry_worst','fractal_dimension_worst'
    ]

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello 👋 I am a Breast Cancer Detector."}
        ]
    if "collecting_features" not in st.session_state:
        st.session_state.collecting_features = False
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "user_features" not in st.session_state:
        st.session_state.user_features = []

    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        reply = ""

        # If user wants to start cancer check
        if user_input.lower().strip() in ["yes i want to check my breast cancer","yes i want to check","check my breast cancer"]:
            st.session_state.collecting_features = True
            st.session_state.current_index = 0
            st.session_state.user_features = []
            reply = f"Okay 👍 Let's start.\nPlease enter **{feature_names[0]}**:"
        
        # If we are collecting features
        elif st.session_state.collecting_features:
            try:
                value = float(user_input)
                st.session_state.user_features.append(value)
                st.session_state.current_index += 1

                if st.session_state.current_index < len(feature_names):
                    reply = f"✅ Got it.\nNow enter **{feature_names[st.session_state.current_index]}**:"
                else:
                    # Prediction
                    prediction = model.predict([st.session_state.user_features])[0]
                    if prediction == 1:
                        reply = "🔴 **Result:** Malignant (Cancerous). Please consult a doctor immediately."
                    else:
                        reply = "🟢 **Result:** Benign (Non-Cancerous)."
                    
                    # Reset state
                    st.session_state.collecting_features = False
                    st.session_state.user_features = []
                    st.session_state.current_index = 0
            except:
                reply = "⚠️ Please enter a valid number."
        
        else:
            # Small talk / normal responses
            if user_input.lower() in ["hi","hello","hi sir","hello sir"]:
                reply = "Hi sir 😊 How can I help you today?"
            elif user_input.lower() in ["who are you","what you can do"]:
                reply = "I am an AI Breast Cancer Detector 🤖. Say *Yes I want to check my breast cancer* to start."
            else:
                reply = "I can help you check breast cancer. Say: **Yes I want to check my breast cancer**"

        # Append reply
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)



with tab2:
    st.subheader("📷 Upload a Breast X-ray (Mammogram)")
    uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img_bytes = uploaded_file.read()

        with st.spinner("🔎 Analyzing image with BreastVision AI..."):
            response = models.generate_content(
                ["You are a medical assistant AI. Analyze this breast X-ray (mammogram) "
                 "and just tell whether it looks more like **benign** or **malignant**.  tell concisely"
                 "Provide Friendly doctor explanantion."
                 "Do not provide medical advice, only an AI-based observation.",
                 {"mime_type": "image/png", "data": img_bytes}]
            )

        st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)
        st.success("✅BreastVision AI Analysis")
        st.write(response.text)
with tab3:
    st.subheader("📊 Model Evaluation Results")

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred, pos_label=1) * 100
    precision = precision_score(y_test, y_pred, pos_label=1) * 100

    st.write(f"**✅ Accuracy:** {accuracy:.2f}%")
    st.write(f"**📌 Precision:** {precision:.2f}%")
    st.write(f"**📌 Recall:** {recall:.2f}%")

    # Classification Report
    st.markdown("### 📑 Classification Report")
    report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
    st.text(report)

    # Confusion Matrix
    st.markdown("### 🔲 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

st.markdown(
    """
    <div style="text-align: center;">
        <h3>BreastVision AI can make mistakes. Check important info </h3>
    </div>
    """,
    unsafe_allow_html=True
)


