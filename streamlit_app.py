import streamlit as st
import os
import io
import uuid
import base64
from datetime import datetime
from PIL import Image

st.set_page_config(
    page_title="Tea Leaf Disease Detection",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

from services.database import init_db, get_session, Prediction
from services.auth import register_user, authenticate_user
from services.ml_service import ml_model, UPLOAD_DIR

init_db()

_bg_path = os.path.join(os.path.dirname(__file__), "static", "BGImage_web.jpg")
_bg_b64 = ""
if os.path.exists(_bg_path):
    with open(_bg_path, "rb") as f:
        _bg_b64 = base64.b64encode(f.read()).decode()

# Primary: warm golden amber     #b58b2b / #c9982e
# Accent:  deep forest green     #2a4a2a / #3d6b3d
# Surface: warm cream            #faf8f4 / #f3efe8
# Text:    warm dark              #2c2418 / #5c5040

_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── Global ──────────────────────────────────────────────── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container {
        max-width: 1100px;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    /* ── Hero banner (auth pages) ────────────────────────────── */
    .hero-banner {
        background: url('data:image/jpeg;base64,%%BG%%') center 60%/cover no-repeat;
        border-radius: 18px;
        padding: 3.5rem 2rem 3rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(160deg,
            rgba(20, 15, 5, 0.85) 0%,
            rgba(45, 35, 15, 0.72) 45%,
            rgba(42, 74, 42, 0.65) 100%);
        border-radius: 18px;
    }
    .hero-content {
        position: relative;
        z-index: 1;
        text-align: center;
    }
    .hero-content h1 {
        color: #f5eedd;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        text-shadow: 0 2px 16px rgba(0, 0, 0, 0.4);
    }
    .hero-content .subtitle {
        color: rgba(245, 238, 221, 0.8);
        font-size: 0.9rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    .hero-content .hero-badge {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.3rem 1rem;
        background: rgba(197, 162, 60, 0.22);
        border: 1px solid rgba(197, 162, 60, 0.4);
        border-radius: 999px;
        color: #e8d48c;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        backdrop-filter: blur(4px);
    }

    /* ── Compact header (predict / history pages) ────────────── */
    .app-header {
        background: url('data:image/jpeg;base64,%%BG%%') center 40%/cover no-repeat;
        border-radius: 14px;
        padding: 1rem 1.5rem;
        margin-bottom: 0.75rem;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg,
            rgba(20, 15, 5, 0.88) 0%,
            rgba(45, 35, 15, 0.76) 50%,
            rgba(42, 74, 42, 0.70) 100%);
        border-radius: 14px;
    }
    .app-header .hero-content h1 { font-size: 1.35rem; }
    .app-header .hero-content .subtitle {
        font-size: 0.78rem;
        margin-top: 0.1rem;
    }

    /* ── Result card ─────────────────────────────────────────── */
    .result-card {
        background: rgba(250, 248, 244, 0.92);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid rgba(181, 139, 43, 0.2);
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(92, 80, 64, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(92, 80, 64, 0.12);
    }

    /* ── Confidence bar ──────────────────────────────────────── */
    .confidence-bar {
        background: #e8e0d0;
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #faf8f4 0%, #f3efe8 100%);
        border-right: 1px solid #e0d8c8;
    }
    [data-testid="stSidebar"] .stButton > button {
        border: none;
        background: transparent;
        color: #5c5040;
        font-weight: 500;
        text-align: left;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(181, 139, 43, 0.1);
        color: #8b6914;
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] .stButton > button:active {
        background: rgba(181, 139, 43, 0.18);
        transform: translateX(3px) scale(0.97);
    }

    /* ── All buttons (global) ────────────────────────────────── */
    .stButton button,
    .stButton > button {
        transition: background 0.3s ease, color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease !important;
        border-radius: 10px;
        font-weight: 600;
        border: 1.5px solid #c4b896 !important;
        color: #4a3c28 !important;
        background: rgba(250, 248, 244, 0.95) !important;
        cursor: pointer;
    }
    .stButton button:hover,
    .stButton > button:hover {
        background: #ede4cc !important;
        border-color: #b58b2b !important;
        color: #5a440a !important;
        box-shadow: 0 2px 10px rgba(181, 139, 43, 0.12) !important;
    }
    .stButton button:active,
    .stButton > button:active {
        background: #e0d6ba !important;
    }

    /* ── Primary / submit buttons ────────────────────────────── */
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #c49a2e, #a87d1c) !important;
        border: none !important;
        color: #fff !important;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 2px 10px rgba(181, 139, 43, 0.3) !important;
        transition: background 0.3s ease, box-shadow 0.3s ease !important;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(135deg, #dab240, #c49a2e) !important;
        box-shadow: 0 4px 18px rgba(181, 139, 43, 0.45) !important;
    }
    [data-testid="stFormSubmitButton"] button:active {
        background: linear-gradient(135deg, #a87d1c, #8c6816) !important;
    }

    /* ── File uploader ───────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #d0c4a8;
        border-radius: 12px;
        padding: 0.75rem;
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #b58b2b;
        background: rgba(181, 139, 43, 0.04);
    }

    /* ── Text inputs (inside forms — light bg, so force dark text) ── */
    [data-testid="stForm"] .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #c4b896 !important;
        padding: 0.6rem 2.5rem 0.6rem 0.85rem !important;
        font-size: 0.92rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        background: #fffdf8 !important;
        color: #3d2e14 !important;
    }
    [data-testid="stForm"] .stTextInput > div > div > input::placeholder {
        color: #a89a80 !important;
        opacity: 1 !important;
    }
    [data-testid="stForm"] .stTextInput > div > div > input:focus {
        border-color: #b58b2b !important;
        box-shadow: 0 0 0 3px rgba(181, 139, 43, 0.15) !important;
        background: #fff !important;
    }

    /* ── Text inputs (outside forms — inherit theme) ───────────── */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 1.5px solid #c4b896 !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #b58b2b !important;
        box-shadow: 0 0 0 3px rgba(181, 139, 43, 0.15) !important;
    }

    /* ── Password field: hide instruction text + pad for eye ────── */
    [data-testid="stForm"] .stTextInput [data-testid="InputInstructions"] {
        display: none !important;
    }
    [data-testid="stForm"] .stTextInput input[type="password"] {
        padding-right: 3rem !important;
    }

    /* ── Form containers (light bg — scoped dark text) ────────── */
    [data-testid="stForm"] {
        border: 1px solid #d4c8a8 !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
        background: #faf7f0 !important;
    }

    /* ── All text inside forms (dark override for light bg) ────── */
    [data-testid="stForm"] [data-testid="stSubheader"],
    [data-testid="stForm"] h3 {
        color: #3d2e14 !important;
        font-weight: 700;
    }
    [data-testid="stForm"] label {
        color: #4a3c28 !important;
        font-weight: 500 !important;
    }
    [data-testid="stForm"] p,
    [data-testid="stForm"] span {
        color: #4a3c28;
    }

    /* ── Info / warning / error boxes ────────────────────────── */
    .stAlert {
        border-radius: 10px !important;
    }

    /* ── Expander styling ────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #5c5040;
    }

    /* ── Hide Streamlit branding ─────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
"""

st.markdown(_css.replace("%%BG%%", _bg_b64), unsafe_allow_html=True)



if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.user_email = None
if "page" not in st.session_state:
    st.session_state.page = "login"




def login_page():
    """Login page with hero banner."""
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-content">
            <h1>Tea Leaf Disease Detection</h1>
            <p class="subtitle">AI-powered tea leaf disease detection with explainable diagnostics</p>
            <span class="hero-badge">EfficientNet &middot; Grad-CAM &middot; v5.1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("Sign In")
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Your password")
            submitted = st.form_submit_button("Sign In", width='stretch', type="primary")

            if submitted:
                if not email or not password:
                    st.error("Please fill in all fields.")
                else:
                    success, message, user_id = authenticate_user(email, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user_id
                        st.session_state.user_email = email
                        st.session_state.page = "predict"
                        st.rerun()
                    else:
                        st.error(message)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Don't have an account? **Register**", width='stretch'):
            st.session_state.page = "register"
            st.rerun()


def register_page():
    """Registration page with hero banner."""
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-content">
            <h1>Tea Leaf Disease Detection</h1>
            <p class="subtitle">Create an account to start diagnosing tea leaf diseases</p>
            <span class="hero-badge">EfficientNet &middot; Grad-CAM &middot; v5.1</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.form("register_form"):
            st.subheader("Create Account")
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Choose a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm")
            submitted = st.form_submit_button("Create Account", width='stretch', type="primary")

            if submitted:
                if not email or not password or not confirm_password:
                    st.error("Please fill in all fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    success, message = register_user(email, password)
                    if success:
                        st.success(message + " Please login.")
                    else:
                        st.error(message)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Already have an account? **Sign In**", width='stretch'):
            st.session_state.page = "login"
            st.rerun()




def sidebar():
    """Render sidebar with navigation."""
    with st.sidebar:
        st.markdown("### Tea Leaf Disease Detection")
        st.markdown(f"Logged in as: **{st.session_state.user_email}**")
        st.markdown("---")

        if st.button("Disease Detection", width='stretch'):
            st.session_state.page = "predict"
            st.rerun()

        if st.button("Prediction History", width='stretch'):
            st.session_state.page = "history"
            st.rerun()

        st.markdown("---")

        if st.button("Logout", width='stretch'):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.session_state.page = "login"
            st.rerun()

        st.markdown("---")
        st.caption("EfficientNet-B3 · Grad-CAM · v5.1")
        st.caption("Algal Leaf · Healthy · Red Leaf Spot · White Spot")


def predict_page():
    """Main prediction page."""
    sidebar()

    st.markdown("""
    <div class="app-header">
        <div class="hero-content">
            <h1>Disease Detection</h1>
            <p class="subtitle">Upload a tea leaf image to instantly identify diseases using AI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns(2, gap="large")

    with col_upload:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a tea leaf image...",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a clear photo of a tea leaf for disease analysis."
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=200)

    with col_result:
        st.markdown("### Diagnostic Results")

        if uploaded_file:
            image_bytes = uploaded_file.getvalue()

            with st.spinner("Analysing leaf patterns..."):
                try:
                    result = ml_model.predict(image_bytes)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    return

            status = result["status"]

            if status == "rejected":
                st.error(result["message"])
                if result.get("crop_display") is not None:
                    st.image(result["crop_display"],
                             caption="Processed Image", width='stretch')
                st.caption(result["disclaimer"])
                return

            confidence_pct = result["confidence"] * 100
            class_name = result["prediction"]
            confidence = result["confidence"]

            # Tier colors (warm palette)
            if status == "high":
                color = "#3d8b3d"       # forest green
                tier_label = "High"
            elif status == "medium":
                color = "#b58b2b"       # golden amber
                tier_label = "Moderate"
            else:
                color = "#c45a3c"       # warm red
                tier_label = "Low"

            st.markdown(f"""
            <div class="result-card">
                <p style="color:#8a7e6c; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.25rem;">
                    Detected Condition
                </p>
                <h2 style="color:{color}; margin:0; font-size:1.7rem; font-weight:800;">
                    {class_name}
                </h2>
                <div style="margin-top:0.8rem;">
                    <div style="display:flex; justify-content:space-between; font-size:0.88rem;">
                        <span style="color:#5c5040; font-weight:500;">Confidence ({tier_label})</span>
                        <span style="color:{color}; font-weight:700;">{confidence_pct:.1f}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{confidence_pct}%; background:{color};"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if status == "low":
                st.warning(result["message"])
            elif status == "medium":
                st.info(result["message"])

            if result.get("all_probs"):
                st.markdown("#### Class Probabilities")
                import pandas as pd
                probs_df = pd.DataFrame(
                    list(result["all_probs"].items()),
                    columns=["Class", "Probability"]
                ).sort_values("Probability", ascending=True)
                st.bar_chart(probs_df.set_index("Class"), horizontal=True)

            st.markdown("#### Explainable AI (Grad-CAM)")
            if result.get("gradcam_overlay") is not None:
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.image(result["crop_display"],
                             caption="Model Input", width='stretch')
                with gcol2:
                    st.image(result["gradcam_overlay"],
                             caption="Grad-CAM Heatmap", width='stretch')
                st.info("The heatmap shows which leaf regions the model focused on. Background activations have been masked out.")
            else:
                st.warning("Heatmap generation unavailable for this prediction.")

            st.caption(result["disclaimer"])

            heatmap_pil = None
            if result.get("gradcam_overlay") is not None:
                from PIL import Image as PILImage
                heatmap_pil = PILImage.fromarray(result["gradcam_overlay"])
            _save_prediction(
                user_id=st.session_state.user_id,
                filename=uploaded_file.name,
                image_bytes=image_bytes,
                class_name=class_name if class_name else "unknown",
                confidence=confidence if confidence else 0.0,
                heatmap_img=heatmap_pil,
            )

        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem 1rem; color:#a89e8c;">
                <p style="font-size:1.1rem; margin-bottom:0.5rem; font-weight:500;">No image uploaded</p>
                <p style="font-size:0.85rem;">Upload a tea leaf photo on the left to see<br>diagnostic results and the AI attribution map.</p>
            </div>
            """, unsafe_allow_html=True)


def _save_prediction(user_id, filename, image_bytes, class_name, confidence, heatmap_img):
    """Save prediction result to the database."""
    try:
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
        saved_filename = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        heatmap_path = None
        if heatmap_img:
            heatmap_filename = f"gradcam_{uuid.uuid4()}.png"
            heatmap_path = os.path.join(UPLOAD_DIR, heatmap_filename)
            heatmap_img.save(heatmap_path)

        db = get_session()
        try:
            prediction = Prediction(
                user_id=user_id,
                filename=filename,
                file_path=file_path,
                heatmap_path=heatmap_path,
                predicted_class=class_name,
                confidence=confidence,
            )
            db.add(prediction)
            db.commit()
        finally:
            db.close()
    except Exception as e:
        print(f"Error saving prediction: {e}")


def history_page():
    """Prediction history page."""
    sidebar()

    st.markdown("""
    <div class="app-header">
        <div class="hero-content">
            <h1>Prediction History</h1>
            <p class="subtitle">Review your past tea leaf disease analyses</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    db = get_session()
    try:
        predictions = (
            db.query(Prediction)
            .filter(Prediction.user_id == st.session_state.user_id)
            .order_by(Prediction.created_at.desc())
            .limit(50)
            .all()
        )
    finally:
        db.close()

    if not predictions:
        st.info("You haven't made any predictions yet. Go to **Disease Detection** to get started!")
        return

    st.markdown(f"Showing **{len(predictions)}** past predictions")
    st.markdown("---")

    for pred in predictions:
        with st.container():
            col_img, col_info = st.columns([1, 3])
            with col_img:
                if pred.file_path and os.path.exists(pred.file_path):
                    st.image(pred.file_path, width=120)
                else:
                    st.markdown("*Image unavailable*")

            with col_info:
                confidence_pct = (pred.confidence or 0) * 100
                is_healthy = pred.predicted_class == "Healthy"
                badge_color = "#3d8b3d" if is_healthy else "#c45a3c"

                st.markdown(f"""
                **{pred.filename}**  
                <span style="background:{badge_color}; color:white; padding:2px 10px; border-radius:12px; font-size:0.85rem;">
                    {pred.predicted_class}
                </span> &nbsp; Confidence: **{confidence_pct:.1f}%**  
                <span style="color:#a89e8c; font-size:0.85rem;">
                    {pred.created_at.strftime('%Y-%m-%d %H:%M') if pred.created_at else 'Unknown'}
                </span>
                """, unsafe_allow_html=True)

                if pred.heatmap_path and os.path.exists(pred.heatmap_path):
                    with st.expander("View Grad-CAM Heatmap"):
                        hcol1, hcol2 = st.columns(2)
                        with hcol1:
                            if os.path.exists(pred.file_path):
                                st.image(pred.file_path, caption="Original", width='stretch')
                        with hcol2:
                            st.image(pred.heatmap_path, caption="Grad-CAM", width='stretch')

            st.markdown("---")




def main():
    """Route to the correct page based on session state."""
    if not st.session_state.authenticated:
        if st.session_state.page == "register":
            register_page()
        else:
            login_page()
    else:
        if st.session_state.page == "history":
            history_page()
        else:
            predict_page()


if __name__ == "__main__":
    main()
