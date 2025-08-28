# -*- coding: utf-8 -*-
import streamlit as st
import os
import subprocess
import tempfile
import sys

# Set proper encoding for Windows
if sys.platform.startswith('win'):
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass
    
    # Force UTF-8 encoding for stdout/stderr
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Extra option: output folder if blend_mode == auto
output_dir = None
output_path = ""

st.set_page_config(page_title="AI Auto Video Editor", layout="wide")
st.title("🎬 AI Auto Video Editor")

# --- Sidebar Controls ---
st.sidebar.header("Analysis Settings")
model_size = st.sidebar.selectbox("Whisper Model Size", ["tiny", "base", "small", "medium", "large"], index=1)
provider = st.sidebar.selectbox("LLM Provider", ["openai", "google"], index=1)
if provider == "openai":
    llm_model = st.sidebar.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o"], index=0)
elif provider == "google":
    llm_model = st.sidebar.selectbox("LLM Model", ["gemini-2.5-flash"], index=0)
blend_mode = st.sidebar.selectbox("Blend Mode", ["auto", "pre-edit", "raw-marker"], index=0)
st.sidebar.text("auto: automatically detects highlights and edits them and outputs edited video in mp4 format")
st.sidebar.text("pre-edit: automatically detects highlights, makes edits, and loads into blender for review")
st.sidebar.text("raw-marker: automatically detects highlights, adds timeline markers, and loads into Blender for review")
test_mode = st.sidebar.checkbox("Test Mode (analysis only)", value=True)

if blend_mode == "auto":
    output_dir = st.sidebar.text_input(
        "Output Folder", value=os.path.abspath("./outputs")
    )
    output_path = os.path.join(output_dir, "highlights.mp4")
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

# --- Additional UI Elements ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Instructions")
st.sidebar.markdown("""
1. Upload a video file or enter the file path
2. Configure analysis settings
3. Click 'Run Auto Editing'
4. Review the results
""")


# --- Video Input Methods ---
input_method = st.radio(
    "Choose input method:",
    ["Upload file", "Enter file path"],
    horizontal=True
)

video_path = None

if input_method == "Upload file":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file:
        # Create a temporary directory for this session
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save uploaded file to temp directory
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.success(f"Video uploaded: {uploaded_file.name}")
        
else:  # Enter file path
    video_path_input = st.text_input(
        "Enter the absolute path to your video file:",
        placeholder="/path/to/your/video.mp4"
    )
    
    if video_path_input:
        video_path = os.path.abspath(video_path_input)
        
        # Validate the file exists
        if os.path.exists(video_path):
            # Check if it's a video file
            valid_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.flv', '.wmv']
            if any(video_path.lower().endswith(ext) for ext in valid_extensions):
                st.success(f"✅ Video file found: {video_path}")
                
                # Optional: Show file info
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                st.info(f"File size: {file_size:.1f} MB")
            else:
                st.error("❌ File doesn't appear to be a supported video format")
                video_path = None
        else:
            st.error("❌ File not found. Please check the path.")
            video_path = None

# Show video preview if available (optional)
if video_path and os.path.exists(video_path):
    #with st.expander("🎥 Video Preview"):
    #    st.video(video_path)

    if video_path and st.button("🚀 Run Auto Editing"):
        with st.spinner("Analyzing and editing video... this may take a while ⏳"):
            # Build CLI command using the current Python interpreter
            cmd = [
                sys.executable,  # This ensures we use the same Python as Streamlit
                "blender_auto_editor.py",
                "--video", video_path,
                "--output", output_path,
                "--model-size", model_size,
                "--blend-mode", blend_mode,
                "--llm-provider", provider,
                "--llm-model", llm_model,
            ]
            if test_mode:
                cmd.append("--test-mode")

            try:
                # Set up environment for UTF-8 support
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['PYTHONUNBUFFERED'] = '1'  # Ensure immediate output

                # Run your script
                process = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
                
                # Display logs
                st.subheader("📋 Pipeline Logs")
                if process.stdout:
                    st.text_area("Output", process.stdout, height=500)
                
                with st.expander("🔧 Errors Info"):
                    if process.stderr:
                        st.text_area("Errors", process.stderr, height=200)
                
                if process.returncode == 0:
                    st.success("Analysis completed successfully! ✅")
                else:
                    st.error(f"Process failed with return code: {process.returncode}")
                    
            except Exception as e:
                st.error(f"Error running the editing pipeline: {str(e)}")

        # Show results if they exist
        if os.path.exists("highlights.txt"):
            st.subheader("⭐ Potential Highlights")
            with open("highlights.txt", "r", encoding="utf-8") as f:
                st.text(f.read())

        # if os.path.exists("edit_points.json"):
        #     st.subheader("📊 Analysis Results")
        #     import json
        #     with open("edit_points.json", "r") as f:
        #         data = json.load(f)
        #         st.json(data)
            
        # Cleanup temp directory only if we created one
        if input_method == "Upload file":
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass  # Ignore cleanup errors



# Show current working directory and Python info for debugging
with st.expander("🔧 Debug Info"):
    st.write("Current working directory:", os.getcwd())
    st.write("Python executable:", sys.executable)
    st.write("Python version:", sys.version)