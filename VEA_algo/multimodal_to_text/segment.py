import torch
import librosa
from pathlib import Path
import os
import json
from scenedetect import detect, ContentDetector, AdaptiveDetector, open_video
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager


def timecode_to_seconds(timecode):
    parts = timecode.split(':')
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def scene_detection(video, detector):
    try:
        video_path = Path(video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video}")

        video_stream = open_video(str(video_path))
        fps = video_stream.frame_rate
        min_scene_len = max(int(fps * 3), 45)

        try:
            video_stream = open_video(str(video_path))
            stats_manager = StatsManager()
            scene_manager = SceneManager(stats_manager)
            scene_manager.add_detector(ContentDetector(threshold=27.0,
                                                       min_scene_len=min_scene_len,
                                                       luma_only=False))
            scene_manager.detect_scenes(video=video_stream)
            scene_list = scene_manager.get_scene_list()

            if not scene_list:
                video_stream = open_video(str(video_path))
                stats_manager = StatsManager()
                scene_manager = SceneManager(stats_manager)
                scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=3.0,     
                                                            min_scene_len=min_scene_len,           
                                                            window_width=2,             
                                                            min_content_val=15.0,       
                                                            luma_only=False))
                scene_manager.detect_scenes(video=video_stream)
                scene_list = scene_manager.get_scene_list()

        except Exception as e:
            print(f"Advanced scene detection failed for {video}, falling back to simple detect: {str(e)}")
            scene_list = detect(str(video_path), detector)

        # Convert scenes to timestamps using full timecode to avoid losing hours/minutes
        starts, ends = [], []
        for scene in scene_list:
            s = timecode_to_seconds(scene[0].get_timecode())
            e = timecode_to_seconds(scene[1].get_timecode())
            starts.append(s)
            ends.append(e)

        if len(starts) == 0:
            starts.append(0.0)
            audio_path = video_path.parent / 'audio.wav'
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            duration = librosa.get_duration(path=str(audio_path))
            ends.append(duration)
            print(f"No scenes detected in {video}, using full duration: {duration}s")

        return (starts, ends)

    except Exception as e:
        print(f"Error in scene detection for {video}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        try:
            audio_path = Path(video).parent / 'audio.wav'
            duration = librosa.get_duration(path=str(audio_path))
            print(f"Falling back to full video duration: {duration}s")
            return ([0.0], [duration])
        except Exception as e2:
            print(f"Error getting audio duration: {str(e2)}")
            print("Using default duration of 0")
            return ([0.0], [0.0])


def segment_with_stt_timestamp(video_dir, starts, ends):

    try:
        with open(video_dir + '/audio.json', 'r', encoding='utf-8', errors='replace') as f:
            stt_result = json.load(f)
    except UnicodeDecodeError:
        with open(video_dir + '/audio.json', 'r', encoding='latin-1') as f:
            stt_result = json.load(f)

    # Support both plain whisper output (dict with 'segments' key)
    # and whisper-diarization output (list of segments with 'speaker' field)
    if isinstance(stt_result, dict) and 'segments' in stt_result:
        segments = stt_result['segments']
        has_speaker = False
    else:
        segments = stt_result
        has_speaker = True

    new_seg = []
    mark = 0
    to_next = False

    for k, (start, end) in enumerate(zip(starts, ends)):
        last_end = start
        sub_seg = {'start': [], 'end': [], 'text': [], 'speaker': []}

        for m, segment in enumerate(segments):
            if m < mark:
                continue
            if segment['end'] < end or to_next or (segment['start'] < end and end - segment['start'] >= segment['end'] - end):
                if segment['start'] - last_end >= 1.0:
                    sub_seg['start'].append(round(last_end, 2))
                    sub_seg['end'].append(round(segment['start'], 2))
                    sub_seg['text'].append('')
                    sub_seg['speaker'].append('')
                    last_end = segment['start']

                sub_seg['start'].append(round(max(segment['start'], start), 2))
                sub_seg['end'].append(round(min(segment['end'], end), 2))
                sub_seg['text'].append(segment['text'])
                sub_seg['speaker'].append(segment.get('speaker', '') if has_speaker else '')
                last_end = segment['end']
                mark += 1
                to_next = False
                continue
            elif segment['start'] < end and end - segment['start'] < segment['end'] - end:
                to_next = True
            if end - last_end >= 1.0:
                sub_seg['start'].append(round(last_end, 2))
                sub_seg['end'].append(round(end, 2))
                sub_seg['text'].append('')
                sub_seg['speaker'].append('')
                last_end = end
            break

        if end - last_end >= 1.0:
            sub_seg['start'].append(round(last_end, 2))
            sub_seg['end'].append(round(end, 2))
            sub_seg['text'].append('')
            sub_seg['speaker'].append('')
            last_end = end

        if len(sub_seg['start']) >= 1:
            new_seg.append(sub_seg)

    return new_seg