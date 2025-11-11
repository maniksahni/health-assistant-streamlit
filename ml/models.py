import os
import pickle
from typing import Optional, Tuple

import streamlit as st


def load_single_model(base_dir: str, model_name: str) -> Optional[object]:
    """Load a single model by name."""
    model_files = {
        "diabetes": "diabetes_model.pkl",
        "heart": "heart_disease_model.pkl",
        "parkinsons": "parkinsons_model.pkl",
    }
    file_name = model_files.get(model_name)
    if not file_name:
        st.error(f"Model '{model_name}' not found.")
        return None

    file_path = os.path.join(base_dir, "saved_models", file_name)
    if not os.path.exists(file_path):
        st.error(f"Model file not found at: {file_path}")
        return None

    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_resource
def load_models(base_dir: str) -> Tuple[object, object, object]:
    dm = load_single_model(base_dir, "diabetes")
    hm = load_single_model(base_dir, "heart")
    pm = load_single_model(base_dir, "parkinsons")
    return dm, hm, pm
