import os
import pickle
from typing import Tuple
import streamlit as st

@st.cache_resource
def load_models(base_dir: str) -> Tuple[object, object, object]:
    with open(os.path.join(base_dir, "saved_models", "diabetes_model.sav"), "rb") as f1:
        dm = pickle.load(f1)
    with open(os.path.join(base_dir, "saved_models", "heart_disease_model.sav"), "rb") as f2:
        hm = pickle.load(f2)
    with open(os.path.join(base_dir, "saved_models", "parkinsons_model.sav"), "rb") as f3:
        pm = pickle.load(f3)
    return dm, hm, pm
