# Turbine: Automated Video Editor 🎬

AI-powered video editing automation tool that identifies highlights in long-form video and implements edits using Blender. This project combines speech recognition, audio analysis, and machine learning to automatically identify exciting moments in videos and generate professional highlight clips.

<img width="2555" height="1268" alt="AI Video Editor - Streamlit" src="https://github.com/user-attachments/assets/49befb0b-c23f-4db4-94b7-3547374cbbd6" />

## ✨ Problem
Editing video content can be a daunting taks. Video editing represents 70-80% of content creation time, with creators manually reviewing hours of footage to identify 2-3 minutes of highlight-worthy moments. This process is both time-intensive and inconsistent, often missing great content buried in long recordings

## 🚀 What It Does

This tool automatically:
1. **Analyzes** your video content using AI
2. **Detects** exciting moments (funny reactions, skill plays, intense gameplay)
3. **Creates** a Blender project with all the highlights
4. **Generates** a professional highlight reel ready for export

Perfect for content creators, gamers, and anyone who wants to quickly turn long videos into engaging highlight content!

https://github.com/user-attachments/assets/4244f43c-ff36-4507-9649-b72fe72f4c59

## 📊 Supported Content Types

- **Gaming Videos**: Skill plays, funny reactions, intense moments
- **Support For Other Content Coming Soon**

## ✨ Features

- **AI-Powered Content Analysis**: Uses OpenAI Whisper for speech recognition and LLM sentiment analysis for moment detection
- **Automatic Highlight Detection**: Identifies moments that are potential highlights
- **Blender Integration**: Seamlessly creates video projects in Blender with proper sequencing
- **Multi-Modal Analysis**: Combines audio features and speech content for comprehensive moment detection
- **Streamlit Web Interface**: User-friendly web UI for easy video upload and processing
- **Customizable Detection**: Configurable keywords and excitement thresholds
- **Cross-Platform Support**: Works on Windows. Limited macOS and Linux support.

## ✨ Challenges


## 📋 Prerequisites

### System Requirements
- **Python 3.8+**
- **Blender 3.0+** (with Python scripting enabled)
- **FFmpeg** installed and accessible in PATH
- **Google AI API Key** (for Gemini integration)

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+ for large videos
- **Storage**: At least 2x the size of your input video for processing
- **GPU**: Optional but highly recommended for faster processing
  - **NVIDIA GPU** with CUDA support (GTX 1060 or better)
  - **CUDA Toolkit** 11.0+ (for GPU acceleration)
  - **cuDNN** compatible with your CUDA version

### Performance Notes
- **CPU-only**: Works but slower, especially with larger Whisper models
- **GPU with CUDA**: Significantly faster processing (2-5x speedup)
- **Model Size Impact**: 
  - `tiny`/`base`: Fast, works well on CPU
  - `small`/`medium`: Benefits from GPU acceleration
  - `large`: GPU highly recommended for reasonable processing times

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/auto-video-editor-blender.git
   cd auto-video-editor-blender
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_ai_api_key_here
   ```

5. **Install FFmpeg** (if not already installed)
   ```bash
   # Windows (with chocolatey):
   choco install ffmpeg
   
   # macOS (with homebrew):
   brew install ffmpeg
   
   # Linux (Ubuntu/Debian):
   sudo apt update && sudo apt install ffmpeg
   ```

6. **Install CUDA (Optional but Recommended)**
   ```bash
   # Windows: Download from NVIDIA website
   # https://developer.nvidia.com/cuda-downloads
   
   # Linux (Ubuntu):
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda
   
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 🎯 Quick Start

### Web Interface (Recommended)
1. **Start the Streamlit app**:
   ```bash
   streamlit run turbine.py
   ```

2. **Open your browser** and go to `http://localhost:8501`

3. **Upload your video** or enter the file path

4. **Configure settings** and click "Run Auto Editing"

## 📖 Usage Examples

### Web Interface Usage
```bash
# Start the Streamlit UI
streamlit run turbine.py

# The web interface will open at http://localhost:8501
# Upload videos, configure settings, and process with a few clicks!
```

## 🔧 Configuration

### Web Interface Settings
- **Whisper Model Size**: Choose from `tiny`, `base`, `small`, `medium`, `large`
- **Blend Mode**: `auto`, `pre-edit`, or `raw-marker`
- **Test Mode**: Analysis only without video generation
- **Output Folder**: Custom output directory for generated files

### Blender Configuration (`blender_config.json`)
- `video_path`: Input video file path
- `output_path`: Output highlight video path
- `edit_points_path`: JSON file containing detected edit points

### Analysis Settings
- **Whisper Model**: Choose from `tiny`, `base`, `small`, `medium`, `large`
- **LLM Model**: Currently supports Google Gemini models
- **Excitement Keywords**: Customizable list of words that indicate exciting moments
- **Confidence Thresholds**: Adjust sensitivity of moment detection

## 🏗️ Project Structure

```
auto-video-editor-blender/
├── turbine.py                 # Streamlit web interface
├── blender_auto_editor.py     # Main analysis and AI processing
├── blend.py                   # Blender project generation
├── blender_test.py            # Testing and validation
├── blender_config.json        # Configuration settings
├── requirements.txt           # Python dependencies
├── edit_points.json           # Generated edit points
├── highlights.txt             # Generated highlights summary
├── summary_stats.txt          # Analysis statistics
├── outputs/                   # Generated output files
├── venv/                      # Virtual environment
└── README.md                  # This file
```

## 🔑 API Keys Required

- **Google AI API**: For Gemini model access
- **Optional**: OpenAI API for alternative LLM models

## 🐛 Troubleshooting

### Common Issues

#### Unicode Encoding Errors (Windows)
If you get `UnicodeEncodeError: 'charmap' codec can't encode character`:
```powershell
# Set environment variables before running
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
streamlit run turbine.py
```

#### Whisper Import Errors
If you get import errors for whisper:
```bash
# Make sure you're in your virtual environment
pip install openai-whisper
```

#### FFmpeg Not Found
```bash
# Windows: Download from https://ffmpeg.org/download.html
# Or install with chocolatey: choco install ffmpeg

# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

#### Blender Integration Issues
- Ensure Blender is installed and accessible in PATH
- Check that Python scripting is enabled in Blender
- Verify the Blender version is 3.0 or higher

#### CUDA/GPU Issues
```bash
# Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check CUDA version
nvidia-smi

# If CUDA is not detected, reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Performance Optimization
- **Use smaller models** (`tiny`, `base`) for CPU-only systems
- **Enable GPU acceleration** for faster processing with larger models
- **Monitor memory usage** - large videos may require more RAM
- **Use SSD storage** for better I/O performance during processing

#### WebSocket Issues on Windows
- **Intermittent socket closure failings**
  
### Getting Help
- Review the configuration examples
- Ensure all dependencies are properly installed

## 📄 License

This project currently has a Portfolio License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **Google Gemini** for AI content analysis
- **Blender Foundation** for the amazing 3D software
- **FFmpeg** for video processing capabilities
- **Streamlit** for the web interface framework
