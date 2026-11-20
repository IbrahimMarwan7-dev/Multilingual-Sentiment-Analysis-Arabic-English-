
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# 1. STREAMLIT CONFIGURATION
st.set_page_config(
    page_title="Advanced BERT Sentiment",
    page_icon="🧠",
    layout="centered"
)

CLASS_NAMES = ["Negative", "Neutral", "Positive"]

MODEL_PATH = Path(
    r"c:\Users\ibrahim\OneDrive - King Salman International University\Level 4\Sem 2\AI based Programming\Project\BERTMODEL"
)

MAX_LENGTH = 128
STRICT_THRESHOLD = 0.75


# 2. LOAD MODEL AND TOKENIZER
@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    use_fast=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        device_map=None,
        low_cpu_mem_usage=False
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# 3. INITIALIZE MODEL
try:

    with st.spinner("Loading BERT Model..."):
        tokenizer, model, device = load_model()

    st.success(f"Model loaded successfully on {device}")

except Exception as e:
    st.error("Failed to load BERT model.")
    st.exception(e)

    st.stop()

# 4. SENTIMENT PREDICTION
def advanced_predict_sentiment(
    text,
    strict_threshold=STRICT_THRESHOLD
):

    # Validate input
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")

    # Tokenization
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    # Move inputs to model device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Model prediction
    with torch.inference_mode():
        outputs = model(**inputs)
        probs = F.softmax(
            outputs.logits,
            dim=-1
        )[0]

    # Extract probabilities
    prob_neg = probs[0].item()
    prob_neu = probs[1].item()
    prob_pos = probs[2].item()

    probabilities = (
        prob_neg,
        prob_neu,
        prob_pos
    )

    # Base prediction
    predicted_class_id = torch.argmax(probs).item()
    base_label = CLASS_NAMES[predicted_class_id]
    base_confidence = probs[predicted_class_id].item()

    # Advanced decision logic
    final_label = base_label
    reason = "High Confidence"

    # Rule 1: Low confidence
    if (
        base_label in ["Positive", "Negative"]
        and base_confidence < strict_threshold
    ):

        final_label = "Neutral"

        reason = "Low Confidence → Neutral"

    # Rule 2: Mixed sentiment
    elif (
        abs(prob_pos - prob_neg) < 0.20
        and prob_neu < 0.40
    ):
        final_label = "Neutral"
        reason = "Mixed Sentiment → Neutral"

    return (
        final_label,
        base_label,
        base_confidence,
        probabilities,
        reason
    )


# 5. UI HELPER FUNCTIONS
def get_color(label):
    colors = {
        "Positive": "#16A34A",
        "Negative": "#DC2626",
        "Neutral": "#64748B"
    }
    return colors.get(label, "#000000")


def plot_probs(probs):
    labels = CLASS_NAMES
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        labels,
        probs,
        color=[
            "#DC2626",
            "#64748B",
            "#16A34A"
        ],
        width=0.5
    )
    ax.set_title(
        "Prediction Probabilities",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.1%}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# 6. MAIN APPLICATION UI
st.title("🧠 Advanced BERT Sentiment Analysis")

st.markdown(
    """
    Analyze text using a fine-tuned BERT model.

    Features:
    - Three-class sentiment classification.
    - Confidence-based decision control.
    - Mixed sentiment detection.
    - Probability visualization.
    - Multiple text analysis.
    """
)

st.divider()

# 7. USER INPUT
user_input = st.text_area(
    "Enter your text:",
    placeholder="Write your review or opinion here...",
    height=150
)

uploaded_file = st.file_uploader(
    "Or upload a TXT file:",
    type=["txt"]
)

# 8. ANALYZE BUTTON
if st.button(
    "🚀 Analyze Sentiment",
    use_container_width=True
):
    # Get input text
    text = ""

    if uploaded_file is not None:
        try:
            file_content = uploaded_file.getvalue()
            text = file_content.decode(
                "utf-8-sig"
            )

        except UnicodeDecodeError:
            st.error(
                "Unable to read the file. "
                "Please upload a UTF-8 encoded TXT file."
            )

            st.stop()

    elif user_input.strip():
        text = user_input

    else:
        st.warning(
            "Please enter text or upload a TXT file."
        )

        st.stop()

    # Split into multiple lines
    texts = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not texts:
        st.warning("No valid text found.")
        st.stop()

    st.subheader("📊 Analysis Results")
    st.caption(
        f"Total texts to analyze: {len(texts)}"
    )

    # Process each text
    for i, current_text in enumerate(texts):
        with st.container(border=True):
            st.markdown(
                f"### 📝 Text {i + 1}"
            )

            st.write(current_text)
            try:
                (
                    label,
                    base_label,
                    conf,
                    probs,
                    reason
                ) = advanced_predict_sentiment(
                    current_text
                )

                # Final prediction
                st.markdown(
                    f"""
                    <h3 style="color:{get_color(label)}">
                        🎯 Final Sentiment: {label}
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

                # Base model prediction
                st.write(
                    f"Base Model Prediction: **{base_label}**"
                )

                # Confidence
                st.markdown("#### 🔍 Model Confidence")
                st.progress(conf)
                st.write(
                    f"Confidence: **{conf * 100:.2f}%**"
                )

                # Probabilities
                st.markdown("#### 📊 Class Probabilities")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Negative",
                        f"{probs[0] * 100:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Neutral",
                        f"{probs[1] * 100:.2f}%"
                    )

                with col3:
                    st.metric(
                        "Positive",
                        f"{probs[2] * 100:.2f}%"
                    )

                # Chart
                plot_probs(probs)

                # Decision explanation
                st.info(
                    f"💡 Decision Reason: {reason}"
                )

            except Exception as e:
                st.error(
                    f"Prediction failed for Text {i + 1}"
                )
                st.exception(e)

st.divider()
st.caption(
    "Advanced BERT Sentiment Analysis | "
    "Powered by PyTorch, Hugging Face & Streamlit"
)
