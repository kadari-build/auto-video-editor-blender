import bpy
from bpy import context, ops
import argparse
import os
from typing import List, Dict
import json
import sys


class BlenderVideoEditor:
    """Handles video editing operations in Blender"""
    
    def __init__(self):
        self.scene = None
        self.sequence_editor = None
        
    def setup_project(self, video_path: str, output_path: str):
        """Initialize Blender project for video editing"""

        print(f"Initializing Blender Project")
            
        # Clear existing data
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Reclaim Scene
        self.scene = bpy.context.scene
        
        # Setup sequence editor
        print(f"Setup Sequence Editor")
        if not self.scene.sequence_editor:
            self.scene.sequence_editor_create()
        self.sequence_editor = self.scene.sequence_editor
        
        # Load video
        print(f"Loading Video")
        #bpy.ops.sequencer.movie_strip_add(
        #    filepath=video_path,
        #    frame_start=1,
        #    channel=1
        #) 
        
        # Add original movie strip directly to sequence editor
        og_strip = self.scene.sequence_editor.strips.new_movie(
        name="orginal_video",
        filepath=video_path,
        channel=1,
        frame_start=1
        )
        print(f"output path: {output_path}")
        # Setup output settings
        print(f"Setting Up Output Settings")
        self.scene.render.filepath = os.path.join(output_path, video_path.split("\\")[-1].split(".")[0] + "_edited.mp4")
        print(f"File Output Path: {self.scene.render.filepath}")
        self.scene.render.fps = int(og_strip.fps)
        self.scene.render.image_settings.file_format = 'FFMPEG'
        self.scene.render.ffmpeg.format = 'MPEG4'
        self.scene.render.ffmpeg.codec = 'H264'
        self.scene.render.ffmpeg.audio_codec = 'MP3'

        return int(og_strip.fps)
        
    def create_highlight_reel(self, edit_points: List[Dict], video_path: str, framerate):
        """Create a highlight reel from edit points"""
            
        print(f"Creating highlight reel with {len(edit_points)} points")
        
        current_frame = 1
        
        for i, point in enumerate(edit_points):
                
            start_frame = int(point['timestamp'] * framerate)
            duration_frames = int(point['duration'] * framerate) + (framerate * 5)
            
            # Add movie clip to timeline
            vid_strip = self.scene.sequence_editor.strips.new_movie(
                name=f"highlight_{i}",
                filepath=video_path,
                channel=2,
                frame_start=current_frame - start_frame
            )

            # Add sound clip to timeline
            audio_strip = self.scene.sequence_editor.strips.new_sound(
                name=f"sound_highlight_{i}",
                filepath=video_path,
                channel=3,
                frame_start=current_frame - start_frame
            )
            
            # Set source range for video file
            vid_strip.frame_offset_start = start_frame
            if vid_strip.frame_offset_start < 0:
                vid_strip.frame_offset_start = 0
            vid_strip.frame_final_duration = duration_frames

            # Set source range for audio file
            audio_strip.frame_offset_start = start_frame
            if audio_strip.frame_offset_start < 0:
                audio_strip.frame_offset_start = 0
            audio_strip.frame_final_duration = duration_frames
            
            # Add transition effect
            if i > 0:
                self._add_transition(current_frame - 15, framerate)
            
            current_frame += duration_frames  # 1 second gap
            
        # Update scene length
        self.scene.frame_end = current_frame

    def create_raw_marker_reel(self, edit_points: List[Dict], video_path: str, framerate):
        """Create a raw marker reel from edit points"""
            
        print(f"Creating Pre-Edit reel with {len(edit_points)} points")
        
        current_frame = 1

        # Add sound clip to timeline
        original_sound = self.scene.sequence_editor.strips.new_sound(
                name=f"original_sound",
                filepath=video_path,
                channel=2,
                frame_start=1
            )
        
        for i, point in enumerate(edit_points):
                
            start_frame = int(point['timestamp'] * framerate)
            duration_frames = int(point['duration'] * framerate)
            
            # Add movie clip to timeline
            marker_start = self.scene.timeline_markers.new(
                name=f"{point['type']}_{i}_start",
                frame=start_frame
            )

            marker_end = self.scene.timeline_markers.new(
                name=f"{point['type']}_{i}_end",
                frame=start_frame + duration_frames
            )
            
            current_frame += duration_frames  # 1 second gap
            
        # Update scene length
        self.scene.frame_end = current_frame
        
    def _add_transition(self, frame_start, duration):
        """Add a simple crossfade transition"""
        # TODO: Implement transition effects
        pass
        
    def render_video(self):
        """Render the final video"""
            
        print("Rendering video...")
        bpy.ops.render.render(animation=True)



def main():
    """Main execution pipeline"""

    print("Blender version:", bpy.app.version_string)
    script_args = sys.argv[(sys.argv.index('--')) + 1:]  # Skip the script name
    print("Arguments passed to script:", script_args)
    
    with open(script_args[0], 'r') as f:
        blender_config = json.load(f)
    
    # Validate input file
    if not os.path.exists(blender_config['video_path']):
        print(f"Error: Video file not found: {blender_config['video_path']}")
        return
    
    print(f"{'='*50}")
    print(f"Blender Video Editor - Starting")
    print(f"Input: {blender_config['video_path']}")
    print(f"Output: {blender_config['output_path']}")
    print(f"Edit Points: {blender_config['edit_points_path']}")
    print(f"{'='*50}")
    
    # Initialize analyzer
    video_editor = BlenderVideoEditor()
    video_framerate = 0
    
    try:       
        print("\n=== Video Editing In Progress ===")
        video_framerate = video_editor.setup_project(blender_config['video_path'], blender_config['output_path'])
        
        # Load edit points
        print("\n= Loading Edit Points =")
        with open(blender_config['edit_points_path'], 'r') as f:
            all_edit_points = json.load(f)
        
        # Filter for highlight-worthy points
        print("\n= Filtering For Highlight Points =")
        highlights = [p for p in all_edit_points['edit_points'] 
           if p['confidence'] >= 0.5 and p['type'] != "silence_removal"]

        print(f"Blender Mode: {script_args[1]}")
        if script_args[1] == 'auto':
            video_editor.create_highlight_reel(highlights, blender_config['video_path'], video_framerate)
            video_editor.render_video()
        elif script_args[1] == 'pre-edit':
            video_editor.create_highlight_reel(highlights, blender_config['video_path'], video_framerate)
        elif script_args[1] == 'raw-marker':
            video_editor.create_raw_marker_reel(highlights, blender_config['video_path'], video_framerate)
                
        print("\n=== Video Editing Completed===")
        print(f"✅ Highlight reel created: {blender_config['output_path']}")  
        
    except Exception as e:
        print(f"❌ Error in video editing pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()