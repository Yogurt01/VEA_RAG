import torch
import numpy as np
import librosa
import cv2
from pathlib import Path
import os
import json
from scenedetect import detect, ContentDetector, AdaptiveDetector, open_video
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager


def get_video_duration(video_path):
    # Get the total duration of the video in seconds.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps


def timecode_to_seconds(timecode):
    # Convert a timecode string (HH:MM:SS.mmm) to total seconds.
    parts = timecode.split(':')
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


# ─────────────────────────────────────────────────────────────────────────────
# STT ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def segment_with_stt_timestamp(video_dir, starts, ends):
    """
    Align transcript (audio.json) with scene boundaries (starts/ends).
    Returns list[dict] with keys: start, end, text, speaker (each element is a scene).
    """
    json_path  = Path(video_dir) / 'audio.json'
    stt_result = None

    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
                stt_result = json.load(f)
        except UnicodeDecodeError:
            try:
                with open(json_path, 'r', encoding='latin-1') as f:
                    stt_result = json.load(f)
            except Exception:
                pass

    if stt_result is None:
        stt_result = []

    if isinstance(stt_result, dict) and 'segments' in stt_result:
        segments   = stt_result['segments']
        has_speaker = False
    else:
        segments    = stt_result
        has_speaker = True

    new_seg = []
    mark    = 0
    to_next = False

    for k, (start, end) in enumerate(zip(starts, ends)):
        last_end = start
        sub_seg  = {'start': [], 'end': [], 'text': [], 'speaker': []}

        for m, segment in enumerate(segments):
            if m < mark:
                continue
            if segment['end'] < end or to_next or (
                segment['start'] < end and
                end - segment['start'] >= segment['end'] - end
            ):
                if segment['start'] - last_end >= 0.05:
                    sub_seg['start'].append(round(last_end, 2))
                    sub_seg['end'].append(round(segment['start'], 2))
                    sub_seg['text'].append('')
                    sub_seg['speaker'].append('')
                    last_end = segment['start']

                sub_seg['start'].append(round(max(segment['start'], start), 2))
                sub_seg['end'].append(round(min(segment['end'], end), 2))
                sub_seg['text'].append(segment['text'])
                sub_seg['speaker'].append(
                    segment.get('speaker', '') if has_speaker else ''
                )
                last_end = segment['end']
                mark    += 1
                to_next  = False
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

        if not sub_seg['start']:
            sub_seg['start'].append(round(start, 2))
            sub_seg['end'].append(round(end, 2))
            sub_seg['text'].append('')
            sub_seg['speaker'].append('')
        new_seg.append(sub_seg)

    return new_seg


# ─────────────────────────────────────────────────────────────────────────────
# SCENE DETECTION  (ContentDetector → AdaptiveDetector → AdaptiveDetector config)
# ─────────────────────────────────────────────────────────────────────────────

def scene_detection(video_path, sensitive_detector=None):
    """
    Run 3 detectors in order, stop as soon as scenes > 3.
    Returns (starts, ends) or (None, None) if no detection.
    """
    video_path = Path(video_path)

    def _run_detector(detector_inst):
        video_stream = open_video(str(video_path))
        stats = StatsManager()
        sm    = SceneManager(stats)
        sm.add_detector(detector_inst)
        sm.detect_scenes(video=video_stream)
        return sm.get_scene_list()

    detectors = [
        ("ContentDetector",       ContentDetector()),
        ("AdaptiveDetector",      AdaptiveDetector()),
        ("AdaptiveDetector(cfg)", sensitive_detector),
    ]

    for label, detector in detectors:
        if detector is None:
            continue
        try:
            scene_list = _run_detector(detector)
            n = len(scene_list) if scene_list else 0
            print(f"  [{label}] → {n} scenes")
            if n > 2:
                starts = [timecode_to_seconds(s[0].get_timecode()) for s in scene_list]
                ends   = [timecode_to_seconds(s[1].get_timecode()) for s in scene_list]
                return starts, ends
        except Exception as e:
            print(f"  [{label}] failed: {e}")

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK METHODS
# ─────────────────────────────────────────────────────────────────────────────

def _motion_boundaries(video_path, sample_fps=5.0, smooth_window=5,
                       threshold_mult=1.8, min_gap_sec=1.5):
    """
    Dense optical flow → local maxima = boundary.
    Suitable for static / talking-head / slow-motion videos.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step       = max(1, int(native_fps / sample_fps))
    magnitudes = []
    prev_gray  = None
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
                magnitudes.append((frame_idx / native_fps, mag))
            prev_gray = gray
        frame_idx += 1

    cap.release()

    if len(magnitudes) < 3:
        return []

    times = np.array([m[0] for m in magnitudes])
    mags  = np.array([m[1] for m in magnitudes])

    if smooth_window > 1 and len(mags) >= smooth_window:
        mags = np.convolve(mags, np.ones(smooth_window)/smooth_window, mode='same')

    threshold  = mags.mean() + threshold_mult * mags.std()
    boundaries = []
    last_t     = -min_gap_sec

    for i in range(1, len(mags) - 1):
        if (mags[i] > mags[i-1] and mags[i] >= mags[i+1]
                and mags[i] > threshold
                and times[i] - last_t >= min_gap_sec):
            boundaries.append(float(times[i]))
            last_t = times[i]

    print(f"  [fallback/motion] → {len(boundaries)} boundaries")
    return boundaries


def _audio_boundaries(video_dir, frame_dur=0.05, smooth_window=10,
                      silence_db=-40.0, min_silence_sec=0.3, min_gap_sec=1.5):
    """
    Silence gaps + energy spikes → boundary.
    Skip if audio.wav is not present.
    """
    audio_path = Path(video_dir) / 'audio.wav'
    if not audio_path.exists():
        print(f"  [fallback/audio] audio.wav not found → skip")
        return []

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    except Exception as e:
        print(f"  [fallback/audio] load failed: {e}")
        return []

    hop    = int(sr * frame_dur)
    rms    = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    times  = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # Silence boundaries
    min_sil_frames   = int(min_silence_sec / frame_dur)
    silence_bds      = []
    last_t           = -min_gap_sec
    i                = 0
    silence_mask     = rms_db < silence_db

    while i < len(silence_mask):
        if silence_mask[i]:
            j = i
            while j < len(silence_mask) and silence_mask[j]:
                j += 1
            if (j - i) >= min_sil_frames:
                mid_t = float(times[(i + j) // 2])
                if mid_t - last_t >= min_gap_sec:
                    silence_bds.append(mid_t)
                    last_t = mid_t
            i = j
        else:
            i += 1

    # Energy spike boundaries
    rms_sm = (np.convolve(rms_db, np.ones(smooth_window)/smooth_window, mode='same')
              if smooth_window > 1 and len(rms_db) >= smooth_window else rms_db)
    diff         = np.diff(rms_sm)
    spike_thr    = diff.mean() + 2.5 * diff.std()
    spike_bds    = []
    last_t       = -min_gap_sec

    for i, d in enumerate(diff):
        if d > spike_thr and times[i] - last_t >= min_gap_sec:
            spike_bds.append(float(times[i]))
            last_t = times[i]

    merged = sorted(set(silence_bds + spike_bds))
    print(f"  [fallback/audio] silence={len(silence_bds)} spike={len(spike_bds)} → {len(merged)} boundaries")
    return merged


def _stt_gap_boundaries(video_dir, min_gap_sec=1.5):
    """
    Gap between two STT segments → boundary at midpoint.
    """
    json_path = Path(video_dir) / 'audio.json'
    if not json_path.exists():
        return []

    try:
        with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [fallback/stt] load failed: {e}")
        return []

    segments   = data.get('segments', data) if isinstance(data, dict) else data
    boundaries = []
    last_t     = -min_gap_sec

    for i in range(1, len(segments)):
        gap_s = segments[i-1].get('end', 0)
        gap_e = segments[i].get('start', 0)
        if gap_e - gap_s >= min_gap_sec:
            mid_t = (gap_s + gap_e) / 2
            if mid_t - last_t >= min_gap_sec:
                boundaries.append(float(mid_t))
                last_t = mid_t

    print(f"  [fallback/stt_gap] → {len(boundaries)} boundaries")
    return boundaries


def _merge_and_ensure_min_nodes(all_boundaries, duration, min_nodes=3,
                                merge_gap=1.0, min_seg_len=1.5):
    """
    1. Sort + remove boundaries too close to start/end
    2. Dedup: remove boundaries closer than merge_gap
    3. If #nodes < min_nodes → add temporal splits to reach count
    Returns (starts, ends).
    """
    bds = sorted(b for b in set(all_boundaries)
                 if min_seg_len < b < duration - min_seg_len)

    deduped = []
    last    = -999.0
    for b in bds:
        if b - last >= merge_gap:
            deduped.append(b)
            last = b

    n_nodes = len(deduped) + 1
    if n_nodes < min_nodes:
        step = duration / min_nodes
        for i in range(1, min_nodes):
            candidate = round(i * step, 3)
            if (min_seg_len < candidate < duration - min_seg_len
                    and all(abs(candidate - b) >= merge_gap for b in deduped)):
                deduped.append(candidate)
        deduped = sorted(deduped)
        n_nodes = len(deduped) + 1

        if n_nodes < min_nodes:
            step    = duration / min_nodes
            deduped = [round(i * step, 3) for i in range(1, min_nodes)]
            print(f"  [merge] hard temporal split → {min_nodes} nodes")

    times  = [0.0] + deduped + [round(duration, 3)]
    starts = [round(t, 3) for t in times[:-1]]
    ends   = [round(t, 3) for t in times[1:]]

    print(f"  [merge] → {len(starts)} nodes (boundaries: {deduped})")
    return starts, ends


def fallback_boundaries(video_path, video_dir, min_nodes=3, merge_gap=1.0):
    """
    Call all fallback signal methods, merge them, return (starts, ends).
    """
    duration = get_video_duration(str(video_path))
    print(f"  [fallback] duration={duration:.2f}s, collecting boundaries...")

    all_bds = []
    all_bds += _motion_boundaries(str(video_path))
    all_bds += _audio_boundaries(str(video_dir))
    all_bds += _stt_gap_boundaries(str(video_dir))

    return _merge_and_ensure_min_nodes(
        all_bds, duration,
        min_nodes=min_nodes,
        merge_gap=merge_gap,
    )