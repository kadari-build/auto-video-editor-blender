import whisper
from whisper.utils import get_writer
import numpy as np
import librosa
import cv2
import json
import os
import tempfile
import argparse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import ffmpeg
import subprocess
from google import genai
from google.genai import types
from openai import OpenAI
import logging
import dotenv
from dotenv import load_dotenv

#load environment variables
load_dotenv()

class EditPoint(BaseModel):
    """Represents a point of interest in the video"""
    timestamp: float
    duration: float
    confidence: float
    event_type: str
    description: str
    audio_features: str
    visual_features: str
    analysis_source: str

class EditPointsList(BaseModel):
    edit_points: List[EditPoint]  # Array inside an object

class AudioAnalyzer:
    """Analyzes audio for interesting moments using Whisper and librosa"""
    
    def __init__(self, whisper_model_size="base", llm_provider="google", llm_model="gemini-2.5-flash"):
        self.whisper_model = whisper.load_model(whisper_model_size)
        self.excitement_keywords = [
            "oh no", "what the", "holy", "wow", "amazing", "incredible",
            "yes!", "no way", "oh my god", "jesus", "shit", "damn",
            "look at", "did you see", "watch out", "help", "run",
            "watch out", "fuck", "died", "no!"
        ]
        self.client = None
        self.llm_config = None
        self.model = llm_model
        self.llm_provider = llm_provider
        self.llm_prompt = f"""
        Analyze this gaming video transcript to find exciting moments to include for a Youtube video:
        
        Find:
        1. Funny moments (jokes, fails, unexpected events)
        2. Skill moments (good plays, clutch moments)
        3. Reaction moments (surprise, excitement, frustration)
        4. Story moments (plot reveals, character interactions)
        5. Action moments (combat, boss fights, intense gameplay)
        
        For each moment, provide:
        - Approximate timestamp
        - Duration of the moment
        - Type of moment
        - A confidence score between 0 and 1 that this is a good moment to include in a Youtube video
        - A description of the moment
        - The source of the analysis (this will default to "LLM")
        
        Return as a list of EditPoint objects with the following fields:
        - timestamp: float
        - duration: float
        - confidence: float
        - event_type: str
        - description: str
        - analysis_source: str
        """
        self.initialize_llm(llm_provider)
    
    def initialize_llm(self, provider: str):
        """Initialize LLM based on provider"""
        
        # Check for required API keys
        if provider == "openai":
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                self.client = OpenAI(api_key=api_key)

            except Exception as e:
                error_msg = str(e)
                if "API key" in error_msg:
                    raise Exception("Invalid or missing API key. Please check your GOOGLE_API_KEY environment variable.")
                elif "model" in error_msg.lower():
                    raise Exception("Failed to initialize Gemini model. Please check if the model is available in your region.")
                else:
                    raise Exception(f"Failed to initialize Gemini: {error_msg}")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found in environment variables")
                return None
        elif provider == "google":
            try:
                api_key = os.getenv("GOOGLE_API_KEY")
                self.client = genai.Client(api_key=api_key)

                self.llm_config = types.GenerateContentConfig(
                    system_instruction=self.llm_prompt,
                    temperature=0.7,
                    response_mime_type="application/json",
                    response_schema=list[EditPoint]
                )
            except Exception as e:
                error_msg = str(e)
                if "API key" in error_msg:
                    raise Exception("Invalid or missing API key. Please check your GOOGLE_API_KEY environment variable.")
                elif "model" in error_msg.lower():
                    raise Exception("Failed to initialize Gemini model. Please check if the model is available in your region.")
                else:
                    raise Exception(f"Failed to initialize Gemini: {error_msg}") 

        
    def analyze_audio(self, video_path: str) -> List[EditPoint]:
        """Main audio analysis pipeline"""
        print(f"Analyzing video: {video_path}")

        
        
        # Whisper can handle video files directly
        print("🎥 Transcribing video...")
        result = self.whisper_model.transcribe(video_path)

        # Specify the output directory and filename
        output_directory = "./"
        output_filename = "transcription.srt"   

        # Get the SRT writer and save the transcription
        srt_writer = get_writer("srt", output_directory)
        srt_writer(result, video_path, {})
        
        # Extract audio for librosa analysis
        print("🎧 Extracting audio for spectral analysis...")
        audio_path = self._extract_audio_for_librosa(video_path)
        
        try:
            # Load audio for acoustic analysis
            y, sr = librosa.load(audio_path)
            
            edit_points = []
            
            # Analyze transcription for keywords
            print("🔍 Detecting keyword moments")
            edit_points.extend(self._detect_keyword_moments(result))

            # Analyze full transcript with LLM
            print("🤖 Analyzing full transcript with LLM")
            edit_points.extend(self._analyze_full_transcript(video_path))
            
            # Analyze acoustic features
            print("🎧 Analyzing acoustic features")
            edit_points.extend(self._detect_volume_spikes(y, sr))
            edit_points.extend(self._detect_laughter_screams(y, sr))
            edit_points.extend(self._detect_silence_gaps(y, sr))

            # Then apply smart boundaries to all points
            print("🔍 Optimizing edit point boundaries")
            edit_points = self._apply_smart_boundaries(edit_points, y, sr, result)
            
            # Sort by timestamp and merge nearby points
            print("🔍 Sorting timeline")
            edit_points.sort(key=lambda x: x.timestamp)
            print(f"👋 Merging moments")
            return edit_points
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return []
            
        finally:
            # Clean up temporary audio file
            print(f"Cleaning up temporary audio file: {audio_path}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    def _extract_audio_for_librosa(self, video_path: str) -> str:
        """Extract audio from video for librosa analysis using ffmpeg-python"""
        # Create temporary audio file
        temp_dir = Path(__file__).parent.absolute()
        audio_filename = f"temp_audio_{os.getpid()}.wav"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        print(f"Extracting audio to: {audio_path}")
        
        try:
            # Extract using ffmpeg-python
            (ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True))
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            raise
        
        return audio_path
    
    def _detect_keyword_moments(self, transcription) -> List[EditPoint]:
        """Detect exciting moments from speech transcription"""
        points = []
        
        for segment in transcription['segments']:
            text = segment['text'].lower()
            
            # Check for excitement keywords
            excitement_score = sum(1 for keyword in self.excitement_keywords 
                                 if keyword in text)
            
            if excitement_score > 0:
                points.append(EditPoint(
                    timestamp=segment['start'],
                    duration=segment['end'] - segment['start'],
                    confidence=min(excitement_score * 0.3, 1.0),
                    event_type="speech_excitement",
                    description=f"Excited speech: {text[:50]}...",
                    audio_features=f"keyword_count: {excitement_score}",
                    visual_features="",
                    analysis_source="Transcription"
                ))
        
        return points
    
    def _detect_volume_spikes(self, y, sr) -> List[EditPoint]:
        """Detect sudden volume increases (shouting, loud events)"""
        points = []
        
        # Calculate RMS energy in windows
        hop_length = sr // 4  # 0.25 second windows
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Find spikes (above 2 standard deviations)
        threshold = np.mean(rms) + 2 * np.std(rms)
        spikes = np.where(rms > threshold)[0]
        
        # Group consecutive spikes
        spike_groups = self._group_consecutive(spikes)
        
        for group in spike_groups:
            start_time = times[group[0]]
            end_time = times[group[-1]]
            max_volume = np.max(rms[group])
            
            points.append(EditPoint(
                timestamp=max(0, start_time - 1),  # Include 1 sec before
                duration=end_time - start_time + 2,  # Include 1 sec after
                confidence=min((max_volume - np.mean(rms)) / np.std(rms) * 0.2, 1.0),
                event_type="volume_spike",
                description=f"Volume spike at {start_time:.1f}s",
                audio_features=f"peak_volume: {float(max_volume)}",
                visual_features="",
                analysis_source="Acoustic"
            ))
        
        return points
    
    def _detect_laughter_screams(self, y, sr) -> List[EditPoint]:
        """Detect laughter and screams using spectral features"""
        points = []
        
        # Analyze in 2-second windows
        window_size = 2 * sr
        hop_size = sr // 2
        
        for i in range(0, len(y) - window_size, hop_size):
            window = y[i:i + window_size]
            timestamp = i / sr
            
            # Spectral features that might indicate laughter/screams
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=window, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(window))
            
            # Heuristic for detecting human vocalizations
            # High frequency content + irregular patterns often = laughter/screams
            if (spectral_centroid > 2000 and 
                zero_crossing_rate > 0.1 and 
                np.std(window) > 0.01):
                
                points.append(EditPoint(
                    timestamp=timestamp,
                    duration=2.0,
                    confidence=0.6,  # Lower confidence, needs refinement
                    event_type="vocalization",
                    description=f"Possible laughter/scream at {timestamp:.1f}s",
                    audio_features=f"spectral_centroid: {float(spectral_centroid)}, zero_crossing_rate: {float(zero_crossing_rate)}",
                    visual_features="",
                    analysis_source="Acoustic"
                ))
        
        return points
    
    def _detect_silence_gaps(self, y, sr) -> List[EditPoint]:
        """Detect long silence periods for removal"""
        points = []
        
        # Find silent regions (below threshold for extended time)
        hop_length = sr // 10
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        silence_threshold = np.mean(rms) * 0.1  # Very quiet
        silent_frames = rms < silence_threshold
        
        # Find continuous silent regions longer than 3 seconds
        silent_groups = self._group_consecutive(np.where(silent_frames)[0])
        
        for group in silent_groups:
            duration = times[group[-1]] - times[group[0]]
            if duration > 3.0:  # Only mark long silences
                points.append(EditPoint(
                    timestamp=times[group[0]],
                    duration=duration,
                    confidence=0.9,
                    event_type="silence_removal",
                    description=f"Long silence ({duration:.1f}s)",
                    audio_features=f"silence_duration: {duration}",
                    visual_features="",
                    analysis_source="Acoustic"
                ))

        return points
    
    def _group_consecutive(self, indices) -> List[List[int]]:
        """Group consecutive indices together"""
        if len(indices) == 0:
            return []
        
        groups = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_group.append(indices[i])
            else:
                groups.append(current_group)
                current_group = [indices[i]]
        
        groups.append(current_group)
        return groups


    def _analyze_full_transcript(self, video_path: str) -> List[EditPoint]:
        """Analyze full transcript with LLM for context and missed moments"""
        
        # Load existing transcript (from your Whisper analysis)
        transcript_file = Path(video_path).stem + ".srt"
        if not Path(transcript_file).exists():
            return []

        llm_points = []
        
        with open(transcript_file, 'r') as f:
            transcript_text = f.read()
        
        try:
            if self.llm_provider == "google":
                response = self.client.models.generate_content(
                    model=self.model,
                    config=self.llm_config,
                    contents=transcript_text
                )
                #print(f"🤖 LLM text response: {json.loads(response.text)}")
                
                # Parse the response and ensure it's a list
                response_data = json.loads(response.text)

                # Convert dictionaries to EditPoint objects if needed
                for point_dict in response_data:
                    if isinstance(point_dict, dict):
                        try:
                            point = EditPoint(**point_dict)
                            llm_points.append(point)
                        except Exception as e:
                            print(f"Warning: Could not create EditPoint from {point_dict}: {e}")
                    elif isinstance(point_dict, EditPoint):
                        llm_points.append(point_dict)
                        
            elif self.llm_provider == "openai":
                response = self.client.responses.parse(
                    model=self.model,
                    instructions=self.llm_prompt,
                    input=transcript_text,
                    text_format=EditPointsList
                )
                response_data = response.output_parsed
                #print(f" LLM edit points: {response_data.edit_points}")
                llm_points = response_data.edit_points

                # Convert dictionaries to EditPoint objects if needed
                # for point_dict in response_data:
                #     if isinstance(point_dict, dict):
                #         try:
                #             point = EditPoint(**point_dict)
                #             llm_points.append(point)
                #         except Exception as e:
                #             print(f"Warning: Could not create EditPoint from {point_dict}: {e}")
                #     elif isinstance(point_dict, EditPoint):
                #         llm_points.append(point_dict)

            return llm_points
                
        except Exception as e:
            print(f"Transcript analysis failed: {e}")
            return []

    def _find_speech_boundaries(self, transcription, target_timestamp, min_duration=2.0):
        """Find natural speech start/end points around a highlight"""
        target_segment = None
        
        # Find the segment containing our highlight
        for segment in transcription['segments']:
            if segment['start'] <= target_timestamp <= segment['end']:
                target_segment = segment
                break
        
        if not target_segment:
            return target_timestamp, target_timestamp + min_duration
        
        # Look for sentence boundaries (periods, exclamations, questions)
        text = target_segment['text']
        words = target_segment.get('words', [])
        
        # Find natural start: beginning of sentence or pause
        start_time = target_segment['start']
        for i, word in enumerate(words):
            if word['start'] <= target_timestamp:
                # Look backwards for sentence start
                for j in range(i, -1, -1):
                    if words[j]['word'].strip().endswith(('.', '!', '?')) and j < len(words) - 1:
                        start_time = words[j + 1]['start']
                        break
                break
        
        # Find natural end: end of sentence or significant pause
        end_time = target_segment['end']
        for word in words:
            if word['start'] >= target_timestamp:
                word_text = word['word'].strip()
                if word_text.endswith(('.', '!', '?')):
                    end_time = word['end'] + 0.5  # Small buffer
                    break
        
        # Ensure minimum duration
        if end_time - start_time < min_duration:
            end_time = start_time + min_duration
        
        return start_time, end_time    


    def _find_energy_boundaries(self, y, sr, target_timestamp, initial_duration):
        """Extend highlight until audio energy decreases significantly"""
        start_sample = int(target_timestamp * sr)
        initial_end_sample = int((target_timestamp + initial_duration) * sr)
        
        # Calculate energy in sliding windows after the initial highlight
        window_size = sr // 4  # 0.25 second windows
        energy_threshold_factor = 0.6  # Energy must drop to 60% of peak
        
        # Find peak energy in the highlight region
        highlight_audio = y[start_sample:initial_end_sample]
        peak_energy = np.max(librosa.feature.rms(y=highlight_audio, hop_length=window_size//4))
        threshold = peak_energy * energy_threshold_factor
        
        # Extend until energy drops below threshold for sustained period
        search_end = min(len(y), initial_end_sample + sr * 10)  # Search up to 10 seconds ahead
        consecutive_low = 0
        required_consecutive = 4  # 1 second of low energy
        
        for i in range(initial_end_sample, search_end, window_size):
            window_energy = np.mean(librosa.feature.rms(y=y[i:i+window_size]))
            
            if window_energy < threshold:
                consecutive_low += 1
                if consecutive_low >= required_consecutive:
                    return target_timestamp, i / sr
            else:
                consecutive_low = 0
        
        # Fallback to original duration + small extension
        return target_timestamp, target_timestamp + initial_duration + 2.0


    def _apply_smart_boundaries(self, edit_points: List[EditPoint], y, sr, transcription) -> List[EditPoint]:
        """Post-process edit points to have smarter boundaries"""
        enhanced_points = []
        try:
            for point in edit_points:
                if point.event_type == "speech_excitement":
                    start, end = self._find_speech_boundaries(transcription, point.timestamp)
                    point.timestamp = start
                    point.duration = end - start
                elif point.event_type in ["volume_spike", "vocalization"]:
                    start, end = self._find_energy_boundaries(y, sr, point.timestamp, point.duration)
                    point.timestamp = start
                    point.duration = end - start
                
                enhanced_points.append(point)
        except Exception as e:
            print(f"Error applying smart boundaries to {point}: {e}")
            raise e
        return enhanced_points

class VisualAnalyzer:
    """Placeholder for visual analysis - hooks for future expansion"""
    
    def __init__(self):
        self.combat_ui_templates = []  # Store UI element templates
        self.scene_change_threshold = 0.3
        
    def analyze_video(self, video_path: str) -> List[EditPoint]:
        """Analyze video for visual cues and enhance existing edit points"""
        print(f"Visual analysis hooks ready for: {video_path}")
        
        # TODO: Implement these methods
        visual_points = []
        
        # Hook: Combat detection
        visual_points.extend(self._detect_combat_sequences(video_path))
        
        # Hook: Scene changes
        visual_points.extend(self._detect_scene_changes(video_path))
        
        # Hook: UI element detection
        visual_points.extend(self._detect_ui_events(video_path))
        
        # Hook: Menu/inventory detection
        visual_points.extend(self._detect_menu_time(video_path))
        
        return visual_points
    
    def _detect_combat_sequences(self, video_path: str) -> List[EditPoint]:
        """TODO: Detect combat by looking for health bars, damage numbers, etc."""
        # Placeholder for combat detection
        return []
    
    def _detect_scene_changes(self, video_path: str) -> List[EditPoint]:
        """TODO: Detect significant scene changes using frame differencing"""
        # Placeholder for scene change detection
        return []
    
    def _detect_ui_events(self, video_path: str) -> List[EditPoint]:
        """TODO: Detect UI events like level ups, achievements, etc."""
        # Placeholder for UI event detection
        return []
    
    def _detect_menu_time(self, video_path: str) -> List[EditPoint]:
        """TODO: Detect when player is in menus/inventory for trimming"""
        # Placeholder for menu detection
        return []

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def merge_nearby_points(points: List[EditPoint], merge_distance=5.0) -> List[EditPoint]:
        """Merge edit points that are close together"""
        if not points:
            return points
        
        merged = [points[0]]
        
        for point in points[1:]:
            last_point = merged[-1]
            
            # If points are close, merge them
            if point.timestamp - (last_point.timestamp + last_point.duration) < merge_distance:
                # Extend the last point to include this one
                new_end = max(last_point.timestamp + last_point.duration, 
                             point.timestamp + point.duration)
                last_point.duration = new_end - last_point.timestamp
                last_point.confidence = max(last_point.confidence, point.confidence)
                last_point.description += f" + {point.description}"
            else:
                merged.append(point)
        
        return merged

def main():
    """Main execution pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Auto Video Editor')
    parser.add_argument('--video', '-v', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', default='./', help='Output directory for files and exported video')
    parser.add_argument('--test-mode', '-t', action='store_true', 
                       help='Test mode - only run analysis, skip Blender editing')
    parser.add_argument('--model-size', '-m', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size')
    parser.add_argument('--blend-mode', '-b', default='auto', 
                       choices=['auto', 'pre-edit', 'raw-marker'],
                       help='Blend mode')
    parser.add_argument('--llm-provider', '-lp', default='google', 
                       choices=['openai', 'google'],
                       help='LLM provider')
    parser.add_argument('--llm-model', '-lm', default='gemini-2.5-flash', 
                       help='LLM model')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    print(f"{'='*50}")
    print(f"🎥 Auto Video Editor - {'Test Mode' if args.test_mode else 'Full Mode'}")
    print(f"Input: {args.video}")
    print(f"Output: {args.output}")
    print(f"Whisper Model: {args.model_size}")
    print(f"Blend Mode: {args.blend_mode}")
    print(f"LLM Provider: {args.llm_provider}")
    print(f"LLM Model: {args.llm_model}")
    print(f"{'='*50}")
    
    # Initialize analyzers
    audio_analyzer = AudioAnalyzer(whisper_model_size=args.model_size, llm_provider=args.llm_provider, llm_model=args.llm_model)
    visual_analyzer = VisualAnalyzer()
    
    try:
        # Step 1: Analyze audio for interesting moments
        print("\n=== Audio Analysis Started===")
        audio_edit_points = audio_analyzer.analyze_audio(args.video)

        if not audio_edit_points:
            print("No audio edit points found. Exiting.")
            raise Exception("No audio edit points found. Exiting.")
        
        print(f"\nFound {len(audio_edit_points)} audio edit points:")
        #for i, point in enumerate(audio_edit_points):
        #    print(f"  {i+1:2d}. {point.timestamp:6.1f}s - {point.event_type:15s} - "
        #          f"conf:{point.confidence:.2f} - {point.description}")

        print("\n=== Audio Analysis Completed===")
        
        # Step 2: Enhance with visual analysis (placeholder)
        print("\n=== Visual Analysis Started===")
        visual_edit_points = visual_analyzer.analyze_video(args.video)
        
        print("\n=== Visual Analysis Completed===")
        
        # Combine all edit points
        all_edit_points = audio_edit_points + visual_edit_points
        all_edit_points.sort(key=lambda x: x.timestamp)
        all_edit_points = merge_nearby_points(all_edit_points)
        
        # Save edit points for review
        edit_data = {
            "video_file": args.video,
            "total_edit_points": len(all_edit_points),
            "edit_points": [
                {
                    "timestamp": p.timestamp,
                    "duration": p.duration,
                    "confidence": p.confidence,
                    "type": p.event_type,
                    "description": p.description,
                    "audio_features": p.audio_features,
                    "visual_features": p.visual_features,
                    "analysis_source": p.analysis_source
                }
                for p in all_edit_points
            ]
        }

        edit_points_path = os.path.join(args.output, "edit_points.json")
        with open(edit_points_path, "w") as f:
            json.dump(edit_data, f, indent=2, default=convert_numpy)
        print(f"\n✅ Edit points saved to: {edit_points_path}")

        # Save edit points for review
        audio_points = {
            "video_file": args.video,
            "total_edit_points": len(audio_edit_points),
            "edit_points": [
                {
                    "timestamp": p.timestamp,
                    "duration": p.duration,
                    "confidence": p.confidence,
                    "type": p.event_type,
                    "description": p.description,
                    "audio_features": p.audio_features,
                    "visual_features": p.visual_features,
                    "analysis_source": p.analysis_source
                }
                for p in audio_edit_points
            ]
        }
        

        audio_points_path = os.path.join(args.output, "audio_edit_points.json")
        with open(audio_points_path, "w") as f:
            json.dump(audio_points, f, indent=2, default=convert_numpy)
        print(f"\n✅ Audio edit points saved to: {audio_points_path}")

         # Create dynamic config for this specific video
        config = {
            'video_path': os.path.abspath(args.video),
            'output_path': os.path.abspath(args.output),
            'edit_points_path': os.path.abspath(edit_points_path)
        }
    
        config_file = os.path.join(args.output, "blender_edit_config.json")
        # Write config for Blender script
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        #Get the config file path
        blender_config_path = os.path.abspath(config_file)
        if args.test_mode:
            print("\n🧪 TEST MODE: Analysis complete. Skipping video editing.")
            print("To run full pipeline, remove --test-mode flag")
            
            # Show summary statistics
            print(f"\n=== Writing Summary Statistics ===")
            event_counts = {}
            for point in all_edit_points:
                event_counts[point.event_type] = event_counts.get(point.event_type, 0) + 1
                
            # Show highlight candidates
            print(f"\n=== Writing Potential Highlights ===")
            highlights = [p for p in all_edit_points 
                         if p.confidence > 0.5 and p.event_type != "silence_removal"]
            
            output_highlights = os.path.join(args.output, "highlights.txt")
            with open(output_highlights, "w", encoding='utf-8') as hf:
                hf.write(f"🎬 Potential Highlights: {len(highlights)} clips\n")
                for point in highlights[:10]:  # Show top 10
                    hf.write(f"  ⭐ {point.timestamp:6.1f}s - {point.description}\n")
                
        else:

             # Show summary statistics
            print(f"\n=== Writing Summary Statistics ===")
            event_counts = {}
            for point in all_edit_points:
                event_counts[point.event_type] = event_counts.get(point.event_type, 0) + 1
                
            # Show highlight candidates
            print(f"\n=== Writing Potential Highlights ===")
            highlights = [p for p in all_edit_points 
                         if p.confidence > 0.5 and p.event_type != "silence_removal"]
            
            output_highlights = os.path.join(args.output, "highlights.txt")
            with open(output_highlights, "w", encoding='utf-8') as hf:
                hf.write(f"🎬 Potential Highlights: {len(highlights)} clips\n")
                for point in highlights[:10]:  # Show top 10
                    hf.write(f"  ⭐ {point.timestamp:6.1f}s - {point.description}\n")

            # Step 3: Create edited video in Blender
            print("\n=== Video Editing Started===")

            if args.blend_mode == 'auto':
                # Call Blender
                subprocess.run(['blender', '--background', '--python', 'blend.py', '--', blender_config_path, args.blend_mode])
            else:
                # Call Blender
                subprocess.run(['blender', '--python', 'blend.py', '--', blender_config_path, args.blend_mode])
        
    except Exception as e:
        print(f"❌ Error in video editing pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()