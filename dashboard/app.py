import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
CLASS_NAMES_PATH = MODELS_DIR / "scene_classes.json"


st.set_page_config(
    page_title="Scene Classifier",
    page_icon="🌄",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(39, 109, 105, .22), transparent 38%),
            #0b1115;
        color: #edf3f2;
    }
    [data-testid="stHeader"] {
        background: rgba(11, 17, 21, .75);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }
    .eyebrow {
        color: #74d3b2;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .subtitle {
        color: #9eaeaf;
        font-size: 1.05rem;
        margin-top: -.5rem;
        margin-bottom: 2rem;
    }
    .panel-title {
        color: #edf3f2;
        font-size: 1.08rem;
        font-weight: 650;
        margin: .3rem 0 .85rem;
    }
    .result-label {
        color: #9eaeaf;
        font-size: .86rem;
        margin-bottom: .35rem;
    }
    .result-class {
        color: #74d3b2;
        font-size: 2.35rem;
        font-weight: 750;
        line-height: 1.15;
        margin-bottom: .7rem;
    }
    div[data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, .025);
        border: 1px dashed rgba(116, 211, 178, .45);
        border-radius: 14px;
    }
    div[data-testid="stProgress"] > div > div {
        background-color: #52c5a1;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, .035);
        border: 1px solid rgba(255, 255, 255, .07);
        border-radius: 12px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def predict_image(image: Image.Image, model, class_names: list[str]) -> list[tuple[str, float]]:
    input_shape = model.input_shape
    height = int(input_shape[1] or 150)
    width = int(input_shape[2] or 150)
    resized_image = image.convert("RGB").resize((width, height))
    image_array = np.asarray(resized_image, dtype=np.float32)[np.newaxis, ...]
    probabilities = np.asarray(model.predict(image_array, verbose=0))[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            f"The model returns {len(probabilities)} scores, but "
            f"{len(class_names)} class names are configured."
        )

    ranked_predictions = sorted(
        zip(class_names, probabilities.tolist()),
        key=lambda prediction: prediction[1],
        reverse=True,
    )
    return ranked_predictions


st.markdown('<div class="eyebrow">Vision AI · Intel scene dataset</div>', unsafe_allow_html=True)
st.title("Scene Classifier")
st.markdown(
    '<div class="subtitle">Upload a photo and let the model identify the scene.</div>',
    unsafe_allow_html=True,
)

model_paths = sorted(
    {
        *MODELS_DIR.glob("*.keras"),
        *MODELS_DIR.glob("*.h5"),
        *MODELS_DIR.glob("*.hdf5"),
    },
    key=lambda path: path.name.lower(),
)

if not model_paths:
    st.error(f"No Keras model was found in `{MODELS_DIR}`.")
    st.stop()

with st.sidebar:
    st.markdown("## Classifier setup")
    selected_model = st.selectbox(
        "Choose a model",
        model_paths,
        format_func=lambda path: path.name,
    )
    st.caption("The model is loaded from the `models` folder.")
    st.divider()
    st.markdown("**Available scene classes**")
    st.caption("Buildings · Forest · Glacier · Mountain · Sea · Street")

try:
    with st.spinner("Loading classifier..."):
        model = load_model(str(selected_model))
except (OSError, ValueError, TypeError, RuntimeError) as exc:
    st.error(f"Could not load `{selected_model.name}`: {exc}")
    st.stop()

try:
    with CLASS_NAMES_PATH.open(encoding="utf-8") as class_file:
        class_names = json.load(class_file)
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"Could not read class names from `{CLASS_NAMES_PATH}`: {exc}")
    st.stop()

if not isinstance(class_names, list) or not class_names or not all(
    isinstance(class_name, str) for class_name in class_names
):
    st.error("The class names file must contain a non-empty list of class names.")
    st.stop()

if model.output_shape[-1] != len(class_names):
    st.error(
        f"`{selected_model.name}` has {model.output_shape[-1]} outputs, but "
        f"`scene_classes.json` contains {len(class_names)} classes."
    )
    st.stop()

st.success(f"Model ready · {selected_model.name}", icon="✅")

upload_column, result_column = st.columns([1.05, 0.95], gap="large")

with upload_column:
    st.markdown('<div class="panel-title">01 &nbsp; Choose an image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop an image here or browse your files",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, PNG and WebP.",
    )

with result_column:
    st.markdown('<div class="panel-title">02 &nbsp; Prediction</div>', unsafe_allow_html=True)
    if uploaded_file is None:
        st.info("Your prediction will appear here after you upload an image.")
    else:
        try:
            image = Image.open(uploaded_file)
            image.load()
            predictions = predict_image(image, model, class_names)
        except (UnidentifiedImageError, OSError, ValueError, TypeError, RuntimeError) as exc:
            st.error(f"Could not classify this image: {exc}")
        else:
            predicted_class, confidence = predictions[0]
            st.markdown('<div class="result-label">PREDICTED SCENE</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-class">{predicted_class.title()}</div>',
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{confidence:.1%}")

            with st.expander("See all class probabilities", expanded=True):
                for class_name, probability in predictions:
                    st.markdown(f"**{class_name.title()}** · {probability:.1%}")
                    st.progress(float(np.clip(probability, 0.0, 1.0)))

            st.image(image, caption=uploaded_file.name, use_container_width=True)
