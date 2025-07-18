import gradio as gr
import pandas as pd
import joblib
model = joblib.load("HeartDiseasePredictor.pkl")
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    input_df = pd.DataFrame(input_features)
    prediction = model.predict(input_df)[0]  
    return "Has Heart Disease" if prediction == 1 else "No Heart Disease"
iface = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Age", value=50, minimum=20, maximum=100),
        gr.Radio([0, 1], label="Sex (0 = Female, 1 = Male)"),
        gr.Radio([0, 1, 2, 3], label="Chest Pain Type (0-3)"),
        gr.Number(label="Resting Blood Pressure", value=120, minimum=80, maximum=200),
        gr.Number(label="Cholesterol", value=200, minimum=100, maximum=600),
        gr.Radio([0, 1], label="Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)"),
        gr.Radio([0, 1, 2], label="Resting ECG Results (0-2)"),
        gr.Number(label="Max Heart Rate Achieved", value=150, minimum=50, maximum=220),
        gr.Radio([0, 1], label="Exercise Induced Angina (0 = No, 1 = Yes)"),
        gr.Number(label="ST Depression", value=1.0, minimum=0.0, maximum=6.0, step=0.1),
        gr.Radio([0, 1, 2], label="Slope of ST Segment (0-2)"),
        gr.Number(label="Number of Major Vessels (0-4)", value=1, minimum=0, maximum=4),
        gr.Radio([0, 1, 2, 3], label="Thalassemia (0-3)"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Heart Disease Predictor",
    description="Enter the patient's details to predict whether they have heart disease or not."
)
if __name__ == "__main__":
    iface.launch()