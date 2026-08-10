"""
insect_tracker.py
-----------------
Tracks dark insect silhouettes in IR-backlit chamber video.
Uses background subtraction + blob detection for detection,
and the Hungarian algorithm for identity-preserving assignment.

Dependencies: opencv-python, scipy, numpy
    pip install opencv-python scipy numpy
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrackerConfig:
    # Crop: fraction of frame width to remove from each side (left & right)
    # Adjust based on your chamber setup — from the example frame ~0.18 works.
    crop_left_frac: float = 0.18
    crop_right_frac: float = 0.18

    # Background subtraction
    bg_history: int = 200          # frames used to build background model
    bg_var_threshold: float = 16   # sensitivity; lower = more sensitive

    # Blob filtering (in pixels, after crop)
    min_area: int = 15             # ignore noise smaller than this
    max_area: int = 800            # ignore large artifacts / clumps
    min_circularity: float = 0.0   # 0.0 = no constraint; raise to filter elongated debris

    # Tracker assignment
    max_distance: float = 60.0     # px — detections further than this start a new track
    max_missing_frames: int = 10   # frames a track can go undetected before being dropped

    # Output
    draw_trails: bool = True
    trail_length: int = 40         # how many past positions to draw
    output_video: bool = True      # write annotated output video


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    positions: list = field(default_factory=list)   # (cx, cy) history
    missing_for: int = 0                            # consecutive frames unseen

    @property
    def last_pos(self):
        return self.positions[-1]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def crop_frame(frame: np.ndarray, cfg: TrackerConfig):  # -> tuple[np.ndarray, int]
    """Remove left/right portions of the frame. Returns (cropped, x_offset)."""
    h, w = frame.shape[:2]
    x0 = int(w * cfg.crop_left_frac)
    x1 = int(w * (1 - cfg.crop_right_frac))
    return frame[:, x0:x1], x0


def detect_insects(
    frame_gray: np.ndarray,
    bg_subtractor: cv2.BackgroundSubtractor,
    cfg: TrackerConfig,
):  # -> list[tuple[int, int]]
    """
    Returns a list of (cx, cy) centroids in the cropped frame's coordinate space.

    Strategy:
      1. Apply background subtraction to isolate moving regions.
      2. Threshold the *original* frame to find dark blobs (the silhouettes).
      3. AND the two masks so we keep only dark things that are also moving.
      4. Filter blobs by area (and optionally circularity).
    """
    # --- background subtraction mask (moving regions) ---
    fg_mask = bg_subtractor.apply(frame_gray)
    # Clean up noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # --- dark-blob mask (insects are darker than the bright background) ---
    # Invert so insects become white; threshold fairly generously.
    _, dark_mask = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- combine ---
    combined = cv2.bitwise_and(fg_mask, dark_mask)

    # Small morphological closing to merge fragmented blobs
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # --- find contours and extract centroids ---
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (cfg.min_area <= area <= cfg.max_area):
            continue

        if cfg.min_circularity > 0:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < cfg.min_circularity:
                continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))

    return centroids


# ---------------------------------------------------------------------------
# Hungarian-algorithm assignment
# ---------------------------------------------------------------------------

# def assign_detections_to_tracks(
#     tracks: list[Track],
#     detections: list[tuple[int, int]],
#     cfg: TrackerConfig,
# ):  # -> tuple[dict[int, int], list[int], list[int]]
def assign_detections_to_tracks(tracks, detections, cfg):  # -> tuple[dict[int, int], list[int], list[int]]
    """
    Returns:
      matches      : {track_index: detection_index}
      unmatched_tracks : list of track indices with no detection this frame
      unmatched_dets   : list of detection indices that don't match any track
    """
    if not tracks or not detections:
        return {}, list(range(len(tracks))), list(range(len(detections)))

    track_pos = np.array([t.last_pos for t in tracks], dtype=float)   # (T, 2)
    det_pos   = np.array(detections,                  dtype=float)     # (D, 2)

    # Cost matrix: Euclidean distance
    diff = track_pos[:, None, :] - det_pos[None, :, :]  # (T, D, 2)
    cost = np.linalg.norm(diff, axis=2)                  # (T, D)

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = {}
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets   = set(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= cfg.max_distance:
            matches[r] = c
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

    return matches, list(unmatched_tracks), list(unmatched_dets)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

COLORS = [
    (255, 80,  80),  (80, 200, 80),  (80, 80, 255),  (255, 200, 0),
    (0, 200, 255),   (255, 0, 200),  (180, 255, 80),  (255, 140, 0),
]

def color_for_id(track_id: int):   # -> tuple[int, int, int]:
    return COLORS[track_id % len(COLORS)]


def draw_frame(
    vis: np.ndarray,
    tracks,
    detections,
    x_offset,
    cfg,
    frame_idx,
):  #  -> np.ndarray
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Raw detections (small grey dots)
    for cx, cy in detections:
        cv2.circle(vis, (cx + x_offset, cy), 3, (180, 180, 180), -1)

    for track in tracks:
        if track.missing_for > 0:
            continue  # only draw active tracks

        color = color_for_id(track.track_id)
        positions = track.positions[-cfg.trail_length:]
        abs_positions = [(x + x_offset, y) for x, y in positions]

        # Trail
        if cfg.draw_trails and len(abs_positions) > 1:
            for i in range(1, len(abs_positions)):
                alpha = i / len(abs_positions)
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(vis, abs_positions[i - 1], abs_positions[i], faded, 1)

        # Current position
        cx, cy = abs_positions[-1]
        cv2.circle(vis, (cx, cy), 5, color, -1)
        cv2.putText(
            vis, str(track.track_id),
            (cx + 6, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
        )

    cv2.putText(
        vis, f"Frame {frame_idx}  Active tracks: {sum(1 for t in tracks if t.missing_for == 0)}",
        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return vis


# ---------------------------------------------------------------------------
# Main tracking loop
# ---------------------------------------------------------------------------

def track_video(video_path: str, cfg=None):
    """
    Run the full tracking pipeline on a video file.

    Returns a dict mapping track_id -> list of (frame_index, cx, cy) tuples
    (coordinates are in the *original* full-frame space).
    """
    if cfg is None:
        cfg = TrackerConfig()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if cfg.output_video:
        out_path = str(Path(video_path).with_suffix("")) + "_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"Writing annotated video to: {out_path}")

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=cfg.bg_history,
        varThreshold=cfg.bg_var_threshold,
        detectShadows=False,
    )

    tracks: list[Track] = []
    next_id = 0
    all_trajectories: dict[int, list] = defaultdict(list)  # id -> [(frame, cx, cy)]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        cropped, x_offset = crop_frame(gray, cfg)

        detections = detect_insects(cropped, bg_sub, cfg)

        # Assignment
        matches, unmatched_tracks, unmatched_dets = assign_detections_to_tracks(
            tracks, detections, cfg
        )

        # Update matched tracks
        for ti, di in matches.items():
            tracks[ti].positions.append(detections[di])
            tracks[ti].missing_for = 0
            all_trajectories[tracks[ti].track_id].append(
                (frame_idx, detections[di][0] + x_offset, detections[di][1])
            )

        # Age unmatched tracks
        for ti in unmatched_tracks:
            tracks[ti].missing_for += 1

        # Spawn new tracks for unmatched detections
        for di in unmatched_dets:
            new_track = Track(track_id=next_id, positions=[detections[di]])
            tracks.append(new_track)
            all_trajectories[next_id].append(
                (frame_idx, detections[di][0] + x_offset, detections[di][1])
            )
            next_id += 1

        # Prune dead tracks
        tracks = [t for t in tracks if t.missing_for <= cfg.max_missing_frames]

        # Visualise
        vis = draw_frame(frame.copy(), tracks, detections, x_offset, cfg, frame_idx)
        if out is not None:
            out.write(vis)

        # Optional: show live (comment out for headless / batch use)
        # cv2.imshow("Tracker", vis)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames, {next_id} total tracks spawned")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    print(f"\nDone. {frame_idx} frames, {next_id} total tracks spawned.")
    return dict(all_trajectories)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    # video_file = sys.argv[1] if len(sys.argv) > 1 else "chamber_video.mp4"
    video_folder = r'D:/buzzwatch_videos/20260806'
    video_file = r''

    cfg = TrackerConfig(
        crop_left_frac=0.05,
        crop_right_frac=0.05,
        bg_history=200,
        bg_var_threshold=16,
        min_area=15,
        max_area=800,
        max_distance=60.0,
        max_missing_frames=10,
        draw_trails=True,
        trail_length=40,
        output_video=True,
    )

    trajectories = track_video(video_file, cfg)

    # Save trajectories as JSON for downstream analysis
    out_json = str(Path(video_file).with_suffix("")) + "_trajectories.json"
    with open(out_json, "w") as f:
        json.dump(trajectories, f)
    print(f"Trajectories saved to: {out_json}")
