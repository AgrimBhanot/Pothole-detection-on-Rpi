"""
Lightweight ByteTrack-Inspired Multi-Object Tracker — Task 2
=============================================================
Provides persistent IDs across frames, per-object lifecycle statistics,
and structured logging when a tracked object leaves the scene.

Core ByteTrack insight implemented here
-----------------------------------------
Use BOTH high-confidence AND low-confidence detections for association,
but only create NEW tracks from high-confidence detections.

This prevents two common failure modes:
  • A real object whose score drops momentarily (occlusion, lighting) would
    be lost by a naïve tracker — ByteTrack keeps it via the low-conf stage.
  • A noise detection would start a spurious track — ByteTrack avoids this
    by gating track creation on the high-confidence tier.

Reference
---------
Zhang et al. "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box" — ECCV 2022.

RPi5 adaptations
-----------------
• No Kalman filter (too CPU-heavy for Pi); greedy IoU matching instead.
• NumPy vectorised IoU matrix; greedy O(min(T,D)) passes.
"""

import time
import numpy as np
from enum import IntEnum
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Track state machine
# ---------------------------------------------------------------------------

class TrackState(IntEnum):
    TRACKED = 1   # Matched in the most recent update call
    LOST    = 2   # Not matched this frame; retained for re-ID
    REMOVED = 3   # Exceeded grace period; logged and purged


# ---------------------------------------------------------------------------
# Track dataclass
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """
    Single tracked object.

    Maintains a persistent ``track_id`` and accumulates statistics for the
    full duration of visibility.  The ``label`` field carries the semantic
    class name so log messages are human-readable.
    """

    track_id:       int
    bbox:           Tuple[int, int, int, int]   # (x1, y1, x2, y2) pixel coords
    score:          float
    state:          TrackState
    first_seen:     float                        # Unix timestamp
    last_seen:      float                        # Unix timestamp
    max_confidence: float
    label:          str = "Object"              # e.g. "Obstacle" | "Pothole"
    lost_frames:    int = 0                     # Consecutive unmatched frames

    # ── Mutations ────────────────────────────────────────────────────────

    def update(
        self,
        bbox:  Tuple[int, int, int, int],
        score: float,
        ts:    float,
    ) -> None:
        """Apply a successfully matched detection to this track."""
        self.bbox           = bbox
        self.score          = score
        self.last_seen      = ts
        self.max_confidence = max(self.max_confidence, score)
        self.state          = TrackState.TRACKED
        self.lost_frames    = 0

    def mark_lost(self) -> None:
        """Record one additional frame without a match."""
        self.state        = TrackState.LOST
        self.lost_frames += 1

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def duration(self) -> float:
        """Wall-clock seconds between first and last confirmed sighting."""
        return self.last_seen - self.first_seen

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    Lightweight ByteTrack-inspired tracker with two-stage matching.

    Stage 1  High-confidence detections  ↔  TRACKED tracks (IoU ≥ threshold)
    Stage 2  Low-confidence  detections  ↔  unmatched TRACKED tracks (relaxed IoU)
    Stage 3  Unmatched high-confidence   ↔  LOST tracks (re-identification)
    Stage 4  Still-unmatched high-conf   →  spawn NEW tracks

    Tracks that stay in LOST state longer than ``max_lost_frames`` are removed
    and their lifecycle statistics are written to stdout.
    """

    def __init__(
        self,
        label:               str   = "Object",
        max_lost_frames:     int   = 15,
        iou_threshold:       float = 0.35,
        high_conf_threshold: float = 0.60,
        low_conf_threshold:  float = 0.30,
    ) -> None:
        """
        Args:
            label:               Semantic class name shown in log messages.
            max_lost_frames:     Grace period (frames) before a lost track
                                 is removed.
            iou_threshold:       Min IoU for a valid stage-1 / stage-3 match.
            high_conf_threshold: Score boundary for high-confidence tier.
            low_conf_threshold:  Score boundary for low-confidence tier.
        """
        self.label           = label
        self.max_lost_frames = max_lost_frames
        self.iou_threshold   = iou_threshold
        self.high_conf       = high_conf_threshold
        self.low_conf        = low_conf_threshold

        self._tracks:  List[Track] = []
        self._next_id: int         = 1

    # ── Public API ────────────────────────────────────────────────────────

    def update(
        self,
        boxes:  List[Tuple[int, int, int, int]],
        scores: List[float],
    ) -> List[Track]:
        """
        Feed current-frame detections and advance all track states.

        This should be called once per frame for the relevant model
        (general or pothole).  If the model did not run this frame
        (alternating schedule), do NOT call update — this avoids
        incorrectly incrementing lost_frames for every other frame.

        Args:
            boxes:  List of (x1, y1, x2, y2) bounding boxes.
            scores: Confidence scores parallel to ``boxes``.

        Returns:
            List of currently TRACKED ``Track`` objects — use these for
            rendering bounding boxes and IDs on the display frame.
        """
        now  = time.time()
        dets = list(zip(boxes, scores))

        # --- Partition detections by confidence tier ----------------------
        high_dets = [(b, s) for b, s in dets if s >= self.high_conf]
        low_dets  = [(b, s) for b, s in dets if self.low_conf <= s < self.high_conf]

        # Snapshot of currently active / lost tracks before this update
        active = [t for t in self._tracks if t.state == TrackState.TRACKED]
        lost   = [t for t in self._tracks if t.state == TrackState.LOST]

        # --- Stage 1: high-conf dets ↔ active tracks ----------------------
        unmatched_active_idx, unmatched_high = self._match_greedy(
            active, high_dets, now, self.iou_threshold
        )

        # --- Stage 2: low-conf dets ↔ still-unmatched active tracks -------
        active_unmatched = [active[i] for i in unmatched_active_idx]
        still_unmatched_active_idx, _ = self._match_greedy(
            active_unmatched, low_dets, now, threshold=0.20
        )

        # --- Stage 3: unmatched high-conf ↔ lost tracks (re-ID) -----------
        _, remaining_high = self._match_greedy(
            lost, unmatched_high, now, self.iou_threshold
        )

        # --- Stage 4: create new tracks from truly unmatched high-conf ----
        for bbox, score in remaining_high:
            self._tracks.append(
                Track(
                    track_id       = self._next_id,
                    bbox           = bbox,
                    score          = score,
                    state          = TrackState.TRACKED,
                    first_seen     = now,
                    last_seen      = now,
                    max_confidence = score,
                    label          = self.label,
                )
            )
            self._next_id += 1

        # --- Mark unmatched active tracks as LOST -------------------------
        for i in still_unmatched_active_idx:
            active_unmatched[i].mark_lost()

        # --- Advance lost-frame counter for pre-existing LOST tracks ------
        # (Only those that were already LOST before this update and were
        # NOT re-identified in Stage 3.)
        for t in lost:
            if t.state == TrackState.LOST:   # Still lost after Stage 3
                t.mark_lost()                # Increment grace-period counter

        # --- Remove tracks that exceeded the grace period -----------------
        self._purge()

        return [t for t in self._tracks if t.state == TrackState.TRACKED]

    def get_active_tracks(self) -> List[Track]:
        """Return all currently TRACKED objects (convenience accessor)."""
        return [t for t in self._tracks if t.state == TrackState.TRACKED]

    def reset(self) -> None:
        """Clear all tracks (e.g. between video clips)."""
        self._tracks  = []
        self._next_id = 1

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _iou(
        b1: Tuple[int, int, int, int],
        b2: Tuple[int, int, int, int],
    ) -> float:
        """Compute intersection-over-union of two (x1,y1,x2,y2) boxes."""
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def _match_greedy(
        self,
        tracks:     List[Track],
        detections: List[Tuple],
        now:        float,
        threshold:  float,
    ) -> Tuple[List[int], List[Tuple]]:
        """
        Greedy highest-IoU matching of detections to tracks.

        Iteratively selects the (track, detection) pair with the globally
        highest IoU, matches them if IoU ≥ threshold, then zeros out that
        row and column before the next iteration.  O(min(T,D)) passes.

        Matched tracks are updated in-place via ``Track.update()``.

        Returns:
            unmatched_track_indices : indices into ``tracks`` with no match
            unmatched_detections    : ``(bbox, score)`` tuples not consumed
        """
        if not tracks or not detections:
            return list(range(len(tracks))), list(detections)

        # Build [T × D] IoU matrix
        iou = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, (b, _) in enumerate(detections):
                iou[i, j] = self._iou(t.bbox, b)

        matched_t: set = set()
        matched_d: set = set()
        work = iou.copy()

        for _ in range(min(len(tracks), len(detections))):
            best = float(work.max())
            if best < threshold:
                break
            r, c = divmod(int(np.argmax(work)), len(detections))
            tracks[r].update(detections[c][0], detections[c][1], now)
            matched_t.add(r)
            matched_d.add(c)
            work[r, :] = 0.0   # Exclude this track from further matching
            work[:, c] = 0.0   # Exclude this detection from further matching

        unmatched_t = [i for i in range(len(tracks))      if i not in matched_t]
        unmatched_d = [detections[j]                       for j in range(len(detections))
                       if j not in matched_d]
        return unmatched_t, unmatched_d

    def _purge(self) -> None:
        """Remove tracks beyond the grace period and emit lifecycle logs."""
        surviving: List[Track] = []
        for t in self._tracks:
            if t.state == TrackState.LOST and t.lost_frames > self.max_lost_frames:
                t.state = TrackState.REMOVED
                self._emit_lifecycle_log(t)
            else:
                surviving.append(t)
        self._tracks = surviving

    def _emit_lifecycle_log(self, track: Track) -> None:
        """
        Print a structured event log when a track lifecycle ends.

        Output example:
            [Tracker] Encountered Pothole #3 | Duration: 4.2s | Max Confidence: 0.87
        """
        print(
            f"[Tracker] Encountered {track.label} #{track.track_id} | "
            f"Duration: {track.duration:.1f}s | "
            f"Max Confidence: {track.max_confidence:.2f}"
        )
