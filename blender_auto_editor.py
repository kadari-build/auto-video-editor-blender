import whisper
import numpy as np
import librosa
import cv2
import json
import os
import tempfile
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import ffmpeg

# Only import Blender modules if not in test mode
try:
    import bpy
    import bmesh
    from bpy import context, data, ops
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

@dataclass
class EditPoint:
    """Represents a point of interest in the video"""
    timestamp: float
    duration: float
    confidence: float
    event_type: str
    description: str
    audio_features: Dict = None
    visual_features: Dict = None

class AudioAnalyzer:
    """Analyzes audio for interesting moments using Whisper and librosa"""
    
    def __init__(self, model_size="base"):
        self.whisper_model = whisper.load_model(model_size)
        self.excitement_keywords = [
            "oh no", "what the", "holy", "wow", "amazing", "incredible",
            "yes!", "no way", "oh my god", "jesus", "shit", "damn",
            "look at", "did you see", "watch out", "help", "run"
        ]
        
    def analyze_audio(self, video_path: str) -> List[EditPoint]:
        """Main audio analysis pipeline"""
        print(f"Analyzing video: {video_path}")
        
        # Whisper can handle video files directly
        result = self.whisper_model.transcribe(video_path)
        
        # Extract audio for librosa analysis
        audio_path = self._extract_audio_for_librosa(video_path)
        
        try:
            # Load audio for acoustic analysis
            y, sr = librosa.load(audio_path)
            
            edit_points = []
            
            # Analyze transcription for keywords
            edit_points.extend(self._detect_keyword_moments(result))
            
            # Analyze acoustic features
            edit_points.extend(self._detect_volume_spikes(y, sr))
            edit_points.extend(self._detect_laughter_screams(y, sr))
            edit_points.extend(self._detect_silence_gaps(y, sr))
            
            # Sort by timestamp and merge nearby points
            edit_points.sort(key=lambda x: x.timestamp)
            return self._merge_nearby_points(edit_points)
            
        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    def _extract_audio_for_librosa(self, video_path: str) -> str:
        """Extract audio from video for librosa analysis using ffmpeg-python"""
        # Create temporary audio file
        temp_dir = tempfile.gettempdir()
        audio_filename = f"temp_audio_{os.getpid()}.wav"
        audio_path = os.path.join(temp_dir, audio_filename)
        
        print(f"Extracting audio to: {audio_path}")
        
        try:
            # Extract using ffmpeg-python
            (ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le')
                .overwrite_output()
                .run(quiet=True, capture_output=True))
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
                    audio_features={"keyword_count": excitement_score, "text": text}
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
                audio_features={"peak_volume": float(max_volume)}
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
                    audio_features={
                        "spectral_centroid": float(spectral_centroid),
                        "zero_crossing_rate": float(zero_crossing_rate)
                    }
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
                    audio_features={"silence_duration": duration}
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
    
    def _merge_nearby_points(self, points: List[EditPoint], merge_distance=5.0) -> List[EditPoint]:
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

class VisualAnalyzer:
    """Placeholder for visual analysis - hooks for future expansion"""
    
    def __init__(self):
        self.combat_ui_templates = []  # Store UI element templates
        self.scene_change_threshold = 0.3
        
    def analyze_video(self, video_path: str, edit_points: List[EditPoint]) -> List[EditPoint]:
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

class BlenderVideoEditor:
    """Handles video editing operations in Blender"""
    
    def __init__(self):
        if not BLENDER_AVAILABLE:
            print("Warning: Blender not available. Video editing features disabled.")
            return
        self.scene = bpy.context.scene
        self.sequence_editor = None
        
    def setup_project(self, video_path: str, output_path: str):
        """Initialize Blender project for video editing"""
        if not BLENDER_AVAILABLE:
            print("Blender not available - skipping video setup")
            return
            
        # Clear existing data
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        # Setup sequence editor
        if not self.scene.sequence_editor:
            self.scene.sequence_editor_create()
        self.sequence_editor = self.scene.sequence_editor
        
        # Load video
        bpy.ops.sequencer.movie_strip_add(
            filepath=video_path,
            frame_start=1,
            channel=1
        )
        
        # Setup output settings
        self.scene.render.filepath = output_path
        self.scene.render.image_settings.file_format = 'FFMPEG'
        self.scene.render.ffmpeg.format = 'MPEG4'
        self.scene.render.ffmpeg.codec = 'H264'
        
    def create_highlight_reel(self, edit_points: List[EditPoint], fps=30):
        """Create a highlight reel from edit points"""
        if not BLENDER_AVAILABLE:
            print("Blender not available - skipping highlight reel creation")
            return
            
        print(f"Creating highlight reel with {len(edit_points)} points")
        
        current_frame = 1
        
        for i, point in enumerate(edit_points):
            if point.event_type == "silence_removal":
                continue  # Skip silence points for highlights
                
            start_frame = int(point.timestamp * fps)
            duration_frames = int(point.duration * fps)
            
            # Add clip to timeline
            strip = self.sequence_editor.sequences.new_movie(
                name=f"highlight_{i}",
                filepath=self.scene.sequence_editor.sequences[0].filepath,
                channel=2,
                frame_start=current_frame
            )
            
            # Set source range
            strip.frame_offset_start = start_frame
            strip.frame_final_duration = duration_frames
            
            # Add transition effect
            if i > 0:
                self._add_transition(current_frame - 15, 30)
            
            current_frame += duration_frames + 30  # 1 second gap
            
        # Update scene length
        self.scene.frame_end = current_frame
        
    def _add_transition(self, frame_start, duration):
        """Add a simple crossfade transition"""
        # TODO: Implement transition effects
        pass
        
    def render_video(self):
        """Render the final video"""
        if not BLENDER_AVAILABLE:
            print("Blender not available - skipping video render")
            return
            
        print("Rendering video...")
        bpy.ops.render.render(animation=True)

def main():
    """Main execution pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Gaming Video Editor')
    parser.add_argument('--video', '-v', required=True, help='Path to input video file')
    parser.add_argument('--output', '-o', default='highlights.mp4', help='Output video path')
    parser.add_argument('--test-mode', '-t', action='store_true', 
                       help='Test mode - only run analysis, skip Blender editing')
    parser.add_argument('--model-size', '-m', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    print(f"{'='*50}")
    print(f"Gaming Video Editor - {'Test Mode' if args.test_mode else 'Full Mode'}")
    print(f"Input: {args.video}")
    print(f"Output: {args.output}")
    print(f"Whisper Model: {args.model_size}")
    print(f"{'='*50}")
    
    # Initialize analyzers
    audio_analyzer = AudioAnalyzer(model_size=args.model_size)
    visual_analyzer = VisualAnalyzer()
    
    if not args.test_mode:
        video_editor = BlenderVideoEditor()
    
    try:
        # Step 1: Analyze audio for interesting moments
        print("\n=== Audio Analysis ===")
        audio_edit_points = audio_analyzer.analyze_audio(args.video)
        
        print(f"\nFound {len(audio_edit_points)} audio edit points:")
        for i, point in enumerate(audio_edit_points):
            print(f"  {i+1:2d}. {point.timestamp:6.1f}s - {point.event_type:15s} - "
                  f"conf:{point.confidence:.2f} - {point.description}")
        
        # Step 2: Enhance with visual analysis (placeholder)
        print("\n=== Visual Analysis ===")
        visual_edit_points = visual_analyzer.analyze_video(args.video, audio_edit_points)
        
        # Combine all edit points
        all_edit_points = audio_edit_points + visual_edit_points
        all_edit_points.sort(key=lambda x: x.timestamp)
        
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
                    "visual_features": p.visual_features
                }
                for p in all_edit_points
            ]
        }
        
        output_json = "edit_points.json"
        with open(output_json, "w") as f:
            json.dump(edit_data, f, indent=2)
        print(f"\n✅ Edit points saved to: {output_json}")
        
        if args.test_mode:
            print("\n🧪 TEST MODE: Analysis complete. Skipping video editing.")
            print("To run full pipeline, remove --test-mode flag")
            
            # Show summary statistics
            print(f"\n=== Summary ===")
            event_counts = {}
            for point in all_edit_points:
                event_counts[point.event_type] = event_counts.get(point.event_type, 0) + 1
            
            for event_type, count in event_counts.items():
                print(f"  {event_type}: {count}")
                
            # Show highlight candidates
            highlights = [p for p in all_edit_points 
                         if p.confidence > 0.5 and p.event_type != "silence_removal"]
            print(f"\n🎬 Potential highlights: {len(highlights)} clips")
            for point in highlights[:10]:  # Show top 10
                print(f"  ⭐ {point.timestamp:6.1f}s - {point.description}")
                
        else:
            # Step 3: Create edited video in Blender
            print("\n=== Video Editing ===")
            video_editor.setup_project(args.video, args.output)
            
            # Filter for highlight-worthy points
            highlights = [p for p in all_edit_points 
                         if p.confidence > 0.5 and p.event_type != "silence_removal"]
            
            video_editor.create_highlight_reel(highlights)
            video_editor.render_video()
            
            print(f"✅ Highlight reel created: {args.output}")
        
    except Exception as e:
        print(f"❌ Error in video editing pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()