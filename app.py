"""
HistoMoE Streamlit Web Dashboard

Provides an interactive UI to upload a histology patch, select a cancer type,
and visualize the HistoMoE predictions including routing weights and gene expression.

Usage:
    streamlit run app.py
"""

import os
import io
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from histomoe.models.histomoe_model import HistoMoE
from histomoe.data.metadata_utils import CANCER_TYPES
from histomoe.data.transforms import get_transforms

# Configure Streamlit page
st.set_page_config(
    page_title="HistoMoE Dashboard",
    page_icon="🔬",
    layout="wide",
)

# --- Define Constants ---
N_GENES = 250
N_EXPERTS = 5

@st.cache_resource
def load_model():
    """Load a fresh HistoMoE model for demo purposes (untrained)."""
    # In a real scenario, you'd load from checkpoint:
    # return HistoMoE.load_from_checkpoint("outputs/checkpoints/best.ckpt")
    
    st.toast("Loading HistoMoE model...")
    model = HistoMoE(
        backbone="resnet50",
        n_genes=N_GENES,
        n_experts=N_EXPERTS,
        gating_mode="soft",
        pretrained_backbone=False # false for fast loading in demo
    )
    model.eval()
    return model

def process_image(image: Image.Image) -> torch.Tensor:
    """Apply evaluation transforms to the uploaded PIL Image."""
    transform = get_transforms(split="test", patch_size=224)
    # Convert RGBA to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor_img = transform(image)
    return tensor_img.unsqueeze(0) # [1, 3, 224, 224]

def plot_routing_bar_chart(routing_weights: np.ndarray, cancer_types: list):
    """Plot a bar chart of routing weights."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
    bars = ax.bar(cancer_types, routing_weights[0], color=colors)
    ax.set_ylabel("Routing Weight")
    ax.set_title("Expert Routing Distribution")
    ax.set_ylim(0, 1.0)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')
        
    return fig

def main():
    st.title("🔬 HistoMoE Interactive Dashboard")
    st.markdown("""
        **Histology-Guided Mixture-of-Experts for Gene Expression Prediction**
        
        Upload a histology image patch (H&E stain), select the tissue origin (cancer type), and see how HistoMoE routes the patch to its specialized experts to predict spatial gene expression.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Input Parameters")
        
        uploaded_file = st.file_uploader("Upload Histology Patch (PNG/JPG)", type=["png", "jpg", "jpeg"])
        
        cancer_type_str = st.selectbox(
            "Select Tissue Type",
            options=CANCER_TYPES,
            index=0
        )
        cancer_type_id = CANCER_TYPES.index(cancer_type_str)
        
        run_btn = st.button("🚀 Run HistoMoE", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("*Note: This demo uses an untrained model initialization for demonstration purposes since real weights require training on protected datasets.*")
        
    # Main content area
    if uploaded_file is None:
        st.info("👈 Please upload a histology image patch in the sidebar to begin.")
        return
        
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Input Image")
        st.image(image, caption=f"Uploaded Patch ({cancer_type_str})", use_container_width=True)
        
    if run_btn:
        with st.spinner("Processing image and predicting gene expression..."):
            # Load model
            model = load_model()
            
            # Prepare inputs
            img_tensor = process_image(image)
            cancer_id_tensor = torch.tensor([cancer_type_id], dtype=torch.long)
            
            # Run inference
            with torch.no_grad():
                result = model.predict_patches(img_tensor, cancer_id_tensor)
                
            predictions = result["predictions"].numpy()
            routing_weights = result["routing_weights"].numpy()
            dominant_expert = result["dominant_expert"].numpy()[0]
            
        with col2:
            st.subheader("🧩 Expert Routing Analysis")
            st.markdown(f"**Dominant Expert:** {CANCER_TYPES[dominant_expert]}")
            
            fig = plot_routing_bar_chart(routing_weights, CANCER_TYPES)
            st.pyplot(fig)
            
        st.markdown("---")
        st.subheader("🧬 Predicted Gene Expression (Top 20 Genes)")
        
        # Display as a dataframe (mock gene names for demo)
        gene_names = [f"Gene_{i:03d}" for i in range(N_GENES)]
        
        # Sort predictions to show highest expressed genes
        preds_flat = predictions[0]
        top_idx = np.argsort(preds_flat)[::-1][:20]
        
        df = pd.DataFrame({
            "Gene Symbol": [gene_names[i] for i in top_idx],
            "Predicted Expression": preds_flat[top_idx]
        })
        
        st.dataframe(
            df.style.background_gradient(cmap="viridis"),
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()
