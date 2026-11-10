import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans

st.set_page_config(page_title="Enzyme Color & Gradient Analyzer", layout="wide")
st.title("Advanced Enzyme & Microplate Analyzer")
st.write("Upload your test strip or microplate image to analyze color percentages across multiple states.")

st.sidebar.header("Analysis Settings")

num_colors = st.sidebar.slider(
    "Number of Color States", 
    min_value=2, max_value=15, value=6,
    help="Increase this if your plate/strip has many gradual concentration steps."
)

filtering_strength = st.sidebar.slider(
    "Background & Border Filtering", 
    min_value=1, max_value=100, value=50,
    help="Adjusts how aggressively outer backgrounds, white spaces, and dark table shadows are removed. Higher = stricter filtering."
)

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(img_rgb, use_container_width=True)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    low_bound = int(np.interp(filtering_strength, [1, 100], [10, 80]))
    high_bound = int(np.interp(filtering_strength, [1, 100], [250, 225]))

    mask = (gray > low_bound) & (gray < high_bound)
    foreground_pixels = img_rgb[mask]

    if len(foreground_pixels) > 0:
        with st.spinner("Clustering color data..."):
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(foreground_pixels)

            labels = kmeans.labels_
            color_palette = kmeans.cluster_centers_.astype(int)

            pixel_counts = np.bincount(labels)
            total_pixels = len(foreground_pixels)
            color_percentages = (pixel_counts / total_pixels) * 100

            sorted_indices = np.argsort(color_percentages)[::-1]

        with col2:
            st.subheader(f"Color Breakdown ({num_colors} States)")
            st.caption(f"Active Filter Range: Dropping pixels below {low_bound} brightness and above {high_bound} brightness.")
            
            for i in sorted_indices:
                r, g, b = color_palette[i]
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                pct = color_percentages[i]

                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <div style="width: 32px; height: 32px; background-color: {hex_color}; border-radius: 6px; border: 1px solid #ccc; margin-right: 12px;"></div>
                        <div style="font-size: 16px;">
                            <strong>State {i+1}</strong>: {pct:.2f}% <span style="color: #777; font-size: 13px;">(RGB: {r}, {g}, {b})</span>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    else:
        st.error("No pixels detected. Try lowering the 'Background & Border Filtering' slider in the sidebar.")
