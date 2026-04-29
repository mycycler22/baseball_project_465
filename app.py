import gradio as gr
import joblib
import numpy as np
import pandas as pd

# --- Load saved model artifacts (trained in notebook) ---
model = joblib.load("logistic_regression.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

FEATURE_COLS = [
    'runner_1B', 'runner_2B', 'runner_3B',
    'usage_CH', 'usage_CU', 'usage_FC', 'usage_FF', 'usage_FO',
    'usage_FS', 'usage_KC', 'usage_KN', 'usage_SI', 'usage_SL', 'usage_ST',
    'bat_side_R', 'if_fielding_alignment_Standard',
    'if_fielding_alignment_Strategic', 'of_fielding_alignment_Strategic'
]

def predict_pitch(
    runner_1B, runner_2B, runner_3B,
    bat_side, if_alignment, of_alignment,
    usage_CH, usage_CU, usage_FC, usage_FF, usage_FO,
    usage_FS, usage_KC, usage_KN, usage_SI, usage_SL, usage_ST
):
    # Build dummy variables to match training encoding
    bat_side_R = 1 if bat_side == "R" else 0
    if_standard = 1 if if_alignment == "Standard" else 0
    if_strategic = 1 if if_alignment == "Strategic" else 0
    of_strategic = 1 if of_alignment == "Strategic" else 0

    features = np.array([[
        runner_1B, runner_2B, runner_3B,
        usage_CH, usage_CU, usage_FC, usage_FF, usage_FO,
        usage_FS, usage_KC, usage_KN, usage_SI, usage_SL, usage_ST,
        bat_side_R, if_standard, if_strategic, of_strategic
    ]])

    features_scaled = scaler.transform(features)

    prediction_encoded = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[0]

    predicted_pitch = label_encoder.inverse_transform(prediction_encoded)[0]

    prob_dict = {
        label: float(prob)
        for label, prob in zip(label_encoder.classes_, probabilities)
    }

    return predicted_pitch, prob_dict


# --- Gradio UI ---
with gr.Blocks(title="MLB Pitch Predictor") as demo:
    gr.Markdown("# ⚾ MLB Pitch Type Predictor")
    gr.Markdown("Enter the game situation and pitcher tendencies to predict the next pitch type.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Game Situation")
            runner_1B = gr.Checkbox(label="Runner on 1st Base")
            runner_2B = gr.Checkbox(label="Runner on 2nd Base")
            runner_3B = gr.Checkbox(label="Runner on 3rd Base")
            bat_side = gr.Radio(choices=["L", "R"], label="Batter Side", value="R")
            if_alignment = gr.Dropdown(
                choices=["Standard", "Strategic", "Other"],
                label="Infield Alignment",
                value="Standard"
            )
            of_alignment = gr.Dropdown(
                choices=["Standard", "Strategic"],
                label="Outfield Alignment",
                value="Standard"
            )

        with gr.Column():
            gr.Markdown("### Pitcher Usage Rates (%)")
            usage_FF = gr.Slider(0, 100, value=40, label="Four-Seam Fastball (FF)")
            usage_SI = gr.Slider(0, 100, value=10, label="Sinker (SI)")
            usage_FC = gr.Slider(0, 100, value=5,  label="Cutter (FC)")
            usage_SL = gr.Slider(0, 100, value=15, label="Slider (SL)")
            usage_ST = gr.Slider(0, 100, value=5,  label="Sweeper (ST)")
            usage_CU = gr.Slider(0, 100, value=10, label="Curveball (CU)")
            usage_KC = gr.Slider(0, 100, value=0,  label="Knuckle Curve (KC)")
            usage_CH = gr.Slider(0, 100, value=10, label="Changeup (CH)")
            usage_FS = gr.Slider(0, 100, value=3,  label="Splitter (FS)")
            usage_FO = gr.Slider(0, 100, value=1,  label="Forkball (FO)")
            usage_KN = gr.Slider(0, 100, value=0,  label="Knuckleball (KN)")

    predict_btn = gr.Button("Predict Pitch", variant="primary")

    with gr.Row():
        predicted_pitch = gr.Text(label="Predicted Pitch Type")
        probabilities = gr.Label(label="Pitch Probabilities", num_top_classes=5)

    predict_btn.click(
        fn=predict_pitch,
        inputs=[
            runner_1B, runner_2B, runner_3B,
            bat_side, if_alignment, of_alignment,
            usage_CH, usage_CU, usage_FC, usage_FF, usage_FO,
            usage_FS, usage_KC, usage_KN, usage_SI, usage_SL, usage_ST
        ],
        outputs=[predicted_pitch, probabilities]
    )

demo.launch()