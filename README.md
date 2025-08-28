# Auto Video Editor Blender 🎬

An intelligent, AI-powered video editing automation tool that creates highlight reels using Blender. This project combines speech recognition, audio analysis, and machine learning to automatically identify exciting moments in videos and generate professional highlight clips.

## ✨ Features

- **AI-Powered Content Analysis**: Uses OpenAI Whisper for speech recognition and Google Gemini for intelligent moment detection
- **Automatic Highlight Detection**: Identifies exciting moments, funny reactions, skill plays, and intense gameplay
- **Blender Integration**: Seamlessly creates video projects in Blender with proper sequencing and transitions
- **Multi-Modal Analysis**: Combines audio features, speech content, and visual cues for comprehensive moment detection
- **Batch Processing**: Handle multiple videos and generate multiple highlight reels
- **Customizable Detection**: Configurable keywords and excitement thresholds

## 🚀 What It Does

This tool automatically:
1. **Analyzes** your video content using AI
2. **Detects** exciting moments (funny reactions, skill plays, intense gameplay)
3. **Creates** a Blender project with all the highlights
4. **Generates** a professional highlight reel ready for export

Perfect for content creators, gamers, and anyone who wants to quickly turn long videos into engaging highlight content!

## 📋 Prerequisites

- **Python 3.8+**
- **Blender 3.0+** (with Python scripting enabled)
- **FFmpeg** installed and accessible in PATH
- **Google AI API Key** (for Gemini integration)

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

## 🎯 Quick Start

1. **Configure your video settings** in `blender_config.json`:
   ```json
   {
     "video_path": "path/to/your/video.mp4",
     "output_path": "path/to/output/highlights.mp4",
     "edit_points_path": "path/to/edit_points.json"
   }
   ```

2. **Run the automatic analysis**:
   ```bash
   python blender_auto_editor.py --video "path/to/your/video.mp4"
   ```

3. **Generate the Blender project**:
   ```bash
   python blend.py
   ```

4. **Open the generated `.blend` file in Blender** and render your highlights!

## 📖 Usage Examples

### Basic Usage
```bash
# Analyze a video and generate edit points
python blender_auto_editor.py --video "gaming_highlight.mp4"

# Create Blender project from edit points
python blend.py
```

### Advanced Usage
```bash
# Custom analysis with specific model
python blender_auto_editor.py --video "video.mp4" --whisper-model "large" --llm-model "gemini-2.5-flash"

# Batch process multiple videos
python blender_auto_editor.py --batch --input-dir "videos/" --output-dir "highlights/"
```

## 🔧 Configuration

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
├── blender_auto_editor.py    # Main analysis and AI processing
├── blend.py                  # Blender project generation
├── blender_test.py           # Testing and validation
├── blender_config.json       # Configuration settings
├── requirements.txt          # Python dependencies
├── video_editor.js          # Web interface (if applicable)
├── edit_points.json         # Generated edit points
└── README.md                # This file
```

## 🤖 How It Works

1. **Audio Analysis**: Uses Whisper for speech-to-text and librosa for audio feature extraction
2. **Content Intelligence**: Google Gemini analyzes transcripts to identify exciting moments
3. **Moment Detection**: Combines audio cues, speech content, and AI analysis
4. **Blender Automation**: Generates complete Blender projects with proper sequencing
5. **Export Ready**: Creates highlight reels ready for final rendering

## 📊 Supported Content Types

- **Gaming Videos**: Skill plays, funny reactions, intense moments
- **Streaming Content**: Highlight clips, memorable reactions
- **Educational Content**: Key insights, important points
- **Entertainment**: Funny moments, dramatic reactions
- **Sports**: Exciting plays, key moments

## 🔑 API Keys Required

- **Google AI API**: For Gemini model access
- **Optional**: OpenAI API for alternative LLM models

## 🐛 Troubleshooting

### Common Issues
- **Blender not found**: Ensure Blender is installed and accessible in PATH
- **FFmpeg errors**: Install FFmpeg and verify it's in your system PATH
- **API key issues**: Check your `.env` file and API key validity
- **Memory issues**: Use smaller Whisper models for large videos

### Getting Help
- Check the [Issues](https://github.com/yourusername/auto-video-editor-blender/issues) page
- Review the configuration examples
- Ensure all dependencies are properly installed

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest

# Format code
black .

# Lint code
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **Google Gemini** for AI content analysis
- **Blender Foundation** for the amazing 3D software
- **FFmpeg** for video processing capabilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/auto-video-editor-blender/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/auto-video-editor-blender/discussions)
- **Wiki**: [Project Wiki](https://github.com/yourusername/auto-video-editor-blender/wiki)

---

⭐ **Star this repository if you find it helpful!** ⭐

Made with ❤️ for content creators and video editors everywhere.
