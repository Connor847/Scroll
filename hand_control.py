#!/usr/bin/env python3
"""
Hand Puppet — Stage 3: scroll and zoom control, one chosen hand.

The webcam tracks the chosen hand (HAND setting, or press H to cycle).

  SCROLL  engage: index + middle extended and together, ring + pinky curled,
                  thumb clear of the fingertips.
          then:   move the fingers up/down; the page follows the movement
                  (natural direction, like a trackpad). Stop = stop.
          exit:   peace sign, open hand, or hand out of view.

  ZOOM    arm:    peace sign (index + middle in a V, thumb tucked, ring +
                  pinky curled). This opens a short window (ZOOM_ARM_TIMEOUT).
          choose: within the window, either
                    pinch  thumb + index + middle tips together -> ZOOM IN
                    spread thumb + index + middle wide apart    -> ZOOM OUT
                  (ring + pinky curled for both)
          then:   ZOOM IN:  spread the fingers; one Cmd+= per ZOOM_STEP of
                            new spread. Ends when approximately fully
                            extended (spread >= ZOOM_DONE_SPREAD).
                  ZOOM OUT: close the fingers; one Cmd+- per ZOOM_STEP of
                            new closure. Ends when approximately fully
                            pinched (spread <= ZOOM_DONE_PINCH).
                  Opening the hand or losing it also ends the zoom.
          after:  idle, until the next peace sign.

Run:   python3 hand_control.py [--dry-run]
       --dry-run prints the actions instead of sending them to macOS.
Keys in the preview window:  Q quit   P pause/resume   H cycle hand
                             S swap Left/Right labels (if they are backwards)
"""

import math
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, HandLandmarksConnections, RunningMode,
)

# ---------------------------------------------------------------------------
# Tunables
#
# Distances are in "palm units": 1.0 = wrist to the knuckle at the base of
# the middle finger, so thresholds do not depend on distance to the camera.
# ---------------------------------------------------------------------------
# -- which hand --
HAND = "Right"            # "Right", "Left" or "Any". H key cycles at runtime.
SWAP_HANDEDNESS = True    # MediaPipe's labels came out reversed for the
                          # mirrored preview; set False if yours are right
                          # without the swap

# -- pose recognition --
TOGETHER_MAX  = 0.35   # index & middle tips closer than this  -> "together"
APART_MIN     = 0.45   # index & middle tips farther than this -> "apart"
PINCH_RADIUS  = 0.50   # thumb, index, middle tips all within this -> pinched
SCROLL_THUMB_CLEAR = 0.60  # thumb tip at least this far from the index tip
                           # for the SCROLL engage pose (> PINCH_RADIUS)
THUMB_OUT_MIN    = 0.70  # thumb tip farther than this from the index knuckle
                         # -> thumb spread (SPREAD pose). Keep > THUMB_TUCKED_MAX.
THUMB_TUCKED_MAX = 0.60  # thumb tip closer than this to the index knuckle
                         # -> thumb tucked (PEACE pose)
HOLD_FRAMES     = 3    # frames a pose must persist before it counts
ARM_HOLD_FRAMES = 2    # same, for pinch/spread during the zoom arm window

# -- scroll --
SCROLL_GAIN      = 700   # scroll pixels per palm-unit of finger movement
SCROLL_DEADZONE  = 0.008 # per-frame movement below this is ignored (jitter)
SCROLL_SMOOTHING = 0.5   # 0 = raw; higher = smoother but laggier (max ~0.8)
SCROLL_INVERT    = False # set True if the page moves the wrong way

# -- zoom --
ZOOM_ARM_TIMEOUT = 1.0   # seconds after the peace sign to form pinch/spread
ZOOM_STEP        = 0.15  # palm-units of new spread/closure per Cmd+= / Cmd+-
ZOOM_DONE_SPREAD = 1.30  # ZOOM IN ends when spread reaches this (fully extended)
ZOOM_DONE_PINCH  = 0.35  # ZOOM OUT ends when spread drops to this (fully pinched)
ZOOM_INVERT      = False # set True if the directions come out backwards

# -- camera --
CAMERA_INDEX = 0

# ---------------------------------------------------------------------------
# Landmark numbering (same as the JavaScript version).
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

HAND_CHOICES = ("Right", "Left", "Any")


# ---------------------------------------------------------------------------
# Geometry + pose classification
# ---------------------------------------------------------------------------
def to_px(lm, w, h):
    return [(p.x * w, p.y * h) for p in lm]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def extended(pts, tip, pip):
    """Tip farther from the wrist than the middle knuckle = extended."""
    return dist(pts[tip], pts[WRIST]) > dist(pts[pip], pts[WRIST])


def measure(pts):
    """All the numbers the decisions are based on, for one hand."""
    palm = dist(pts[WRIST], pts[MIDDLE_MCP]) or 1.0
    return {
        "palm": palm,
        "idx_ext":  extended(pts, INDEX_TIP,  INDEX_PIP),
        "mid_ext":  extended(pts, MIDDLE_TIP, MIDDLE_PIP),
        "ring_ext": extended(pts, RING_TIP,   RING_PIP),
        "pink_ext": extended(pts, PINKY_TIP,  PINKY_PIP),
        "d_im": dist(pts[INDEX_TIP], pts[MIDDLE_TIP]) / palm,  # index <-> middle
        "d_ti": dist(pts[THUMB_TIP], pts[INDEX_TIP])  / palm,  # thumb <-> index
        "d_tm": dist(pts[THUMB_TIP], pts[MIDDLE_TIP]) / palm,  # thumb <-> middle
        "d_tk": dist(pts[THUMB_TIP], pts[INDEX_MCP])  / palm,  # thumb <-> index knuckle
        # control signals:
        "tip_y":  (pts[INDEX_TIP][1] + pts[MIDDLE_TIP][1]) / 2 / palm,
        "spread": (dist(pts[THUMB_TIP], pts[INDEX_TIP]) +
                   dist(pts[THUMB_TIP], pts[MIDDLE_TIP])) / 2 / palm,
    }


def classify(m):
    """Pose name from a measurement dict:
    OPEN, PINCH, SPREAD, PEACE, SCROLL, or NEUTRAL."""
    if m["ring_ext"] or m["pink_ext"]:
        return "OPEN"
    if m["d_ti"] < PINCH_RADIUS and m["d_tm"] < PINCH_RADIUS and m["d_im"] < PINCH_RADIUS:
        return "PINCH"
    if not (m["idx_ext"] and m["mid_ext"]):
        return "NEUTRAL"
    if m["d_im"] > APART_MIN:
        if m["d_tk"] > THUMB_OUT_MIN:
            return "SPREAD"          # V with thumb out wide
        if m["d_tk"] < THUMB_TUCKED_MAX:
            return "PEACE"           # V with thumb tucked
        return "NEUTRAL"
    if m["d_im"] < TOGETHER_MAX and m["d_ti"] > SCROLL_THUMB_CLEAR:
        return "SCROLL"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Output layers: real macOS events, or dry-run printing.
# ---------------------------------------------------------------------------
class DryRunOutput:
    name = "DRY RUN"

    def scroll(self, dy):
        print(f"scroll {dy:+d}px")

    def zoom(self, direction):
        print("zoom", "IN" if direction > 0 else "OUT")


class MacOutput:
    """Posts events the way a trackpad / keyboard would, via Quartz.
    Needs Accessibility permission for the terminal app."""
    name = "macOS"
    KEY_EQUALS = 24   # '=' : Cmd+= is zoom-in in Safari, Chrome, Preview, ...
    KEY_MINUS  = 27   # '-' : Cmd+-

    def __init__(self):
        import Quartz
        self.Q = Quartz
        self._check_accessibility()

    def _check_accessibility(self):
        try:
            from ApplicationServices import (AXIsProcessTrustedWithOptions,
                                             kAXTrustedCheckOptionPrompt)
            trusted = AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
        except Exception:
            return
        if not trusted:
            print("Accessibility permission is not granted. macOS should have "
                  "opened a prompt; allow your terminal app in System Settings "
                  "> Privacy & Security > Accessibility, then quit and reopen "
                  "the terminal and run again.")
            sys.exit(1)

    def scroll(self, dy):
        Q = self.Q
        ev = Q.CGEventCreateScrollWheelEvent(None, Q.kCGScrollEventUnitPixel, 1, int(dy))
        Q.CGEventPost(Q.kCGHIDEventTap, ev)

    def zoom(self, direction):
        Q = self.Q
        key = self.KEY_EQUALS if direction > 0 else self.KEY_MINUS
        for down in (True, False):
            ev = Q.CGEventCreateKeyboardEvent(None, key, down)
            Q.CGEventSetFlags(ev, Q.kCGEventFlagMaskCommand)
            Q.CGEventPost(Q.kCGHIDEventTap, ev)


# ---------------------------------------------------------------------------
# Controller: per-frame measurements -> actions.
#
# States:  IDLE -> SCROLL  (via SCROLL pose)        -> IDLE (PEACE/OPEN/lost)
#          IDLE -> ARMED   (via PEACE)              -> IDLE (timeout/OPEN/lost)
#          ARMED -> ZOOM_IN  (via PINCH)  -> IDLE when fully extended / OPEN / lost
#          ARMED -> ZOOM_OUT (via SPREAD) -> IDLE when fully pinched  / OPEN / lost
#          SCROLL -> ARMED (via PEACE)
# ---------------------------------------------------------------------------
class Controller:
    def __init__(self, output, clock=time.time):
        self.out = output
        self.clock = clock
        self.state = "IDLE"
        self.candidate, self.count = None, 0   # pose being counted toward HOLD_FRAMES
        self.armed_at = 0.0
        self.prev_y = None
        self.smooth_dy = 0.0      # low-pass filtered scroll movement
        self.zoom_mark = None     # high-water (in) / low-water (out) spread
        self.last_action = ""

    # -- helpers --
    def _held(self, pose, want, frames=None):
        """True once `pose` (in `want`) has been seen `frames` frames running."""
        if frames is None:
            frames = HOLD_FRAMES
        if pose in want:
            self.count = self.count + 1 if pose == self.candidate else 1
            self.candidate = pose
            return self.count >= frames
        self.candidate, self.count = None, 0
        return False

    def _go(self, state, note):
        self.state = state
        self.candidate, self.count = None, 0
        self.last_action = note

    # -- main entry --
    def update(self, m):
        """m = measurement dict for the chosen hand this frame, or None."""
        if m is None:
            if self.state != "IDLE":
                self._go("IDLE", "idle (hand lost)")
            return
        pose = classify(m)
        getattr(self, "_in_" + self.state.lower())(m, pose)

    def _in_idle(self, m, pose):
        if self._held(pose, ("SCROLL", "PEACE")):
            if pose == "SCROLL":
                self.prev_y = m["tip_y"]
                self.smooth_dy = 0.0
                self._go("SCROLL", "scroll engaged")
            else:
                self.armed_at = self.clock()
                self._go("ARMED", "armed: pinch = zoom in, spread = zoom out")

    def _in_scroll(self, m, pose):
        if self._held(pose, ("PEACE", "OPEN")):
            if pose == "PEACE":
                self.armed_at = self.clock()
                self._go("ARMED", "armed: pinch = zoom in, spread = zoom out")
            else:
                self._go("IDLE", "idle (open hand)")
            return
        dy = m["tip_y"] - self.prev_y            # palm units, + = moved down
        self.prev_y = m["tip_y"]
        if abs(dy) < SCROLL_DEADZONE:
            dy = 0.0
        # low-pass filter: blend this frame's movement with the running value
        self.smooth_dy = SCROLL_SMOOTHING * self.smooth_dy + (1 - SCROLL_SMOOTHING) * dy
        dy = self.smooth_dy
        if abs(dy) < SCROLL_DEADZONE / 2:         # settled: stop sending
            return
        # Natural scrolling: fingers up (dy < 0) -> content moves up -> page
        # scrolls toward the bottom, which is a negative wheel value.
        px = int(round(dy * SCROLL_GAIN))
        if SCROLL_INVERT:
            px = -px
        if px:
            self.out.scroll(px)
            self.last_action = f"scroll {px:+d}px"

    def _in_armed(self, m, pose):
        if self.clock() - self.armed_at > ZOOM_ARM_TIMEOUT:
            self._go("IDLE", "idle (arm window expired)")
            return
        if self._held(pose, ("PINCH", "SPREAD", "OPEN"), ARM_HOLD_FRAMES):
            if pose == "PINCH":
                self.zoom_mark = m["spread"]
                self._go("ZOOM_IN", "zoom in: spread the fingers")
            elif pose == "SPREAD":
                self.zoom_mark = m["spread"]
                self._go("ZOOM_OUT", "zoom out: close the fingers")
            else:
                self._go("IDLE", "idle (open hand)")

    def _in_zoom_in(self, m, pose):
        if pose == "OPEN" or m["spread"] >= ZOOM_DONE_SPREAD:
            self._go("IDLE", "idle (zoom in finished)")
            return
        # ratchet: fire a step for every ZOOM_STEP of NEW spread beyond the
        # high-water mark; closing a little does nothing.
        while m["spread"] - self.zoom_mark >= ZOOM_STEP:
            self.zoom_mark += ZOOM_STEP
            self._zoom(+1)

    def _in_zoom_out(self, m, pose):
        if pose == "OPEN" or m["spread"] <= ZOOM_DONE_PINCH:
            self._go("IDLE", "idle (zoom out finished)")
            return
        while self.zoom_mark - m["spread"] >= ZOOM_STEP:
            self.zoom_mark -= ZOOM_STEP
            self._zoom(-1)

    def _zoom(self, direction):
        if ZOOM_INVERT:
            direction = -direction
        self.out.zoom(direction)
        self.last_action = "zoom " + ("IN" if direction > 0 else "OUT")


# ---------------------------------------------------------------------------
# Hand selection
# ---------------------------------------------------------------------------
def label_hands(result, swap):
    """List of (landmarks, label) for every hand seen, label = "Left"/"Right"
    after the optional swap."""
    out = []
    for lms, hd in zip(result.hand_landmarks, result.handedness):
        label = hd[0].category_name
        if swap:
            label = "Left" if label == "Right" else "Right"
        out.append((lms, label))
    return out


def pick_hand(result, want, swap=None):
    """Return (landmarks, label) for the hand matching `want`, or (None, seen)
    where `seen` lists the labels that were visible."""
    if swap is None:
        swap = SWAP_HANDEDNESS
    hands = label_hands(result, swap)
    for lms, label in hands:
        if want == "Any" or label == want:
            return lms, label
    return None, [label for _, label in hands]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
STATE_COLOR = {
    "IDLE": (160, 160, 160), "SCROLL": (255, 229, 0), "ARMED": (0, 200, 255),
    "ZOOM_IN": (80, 220, 80), "ZOOM_OUT": (80, 120, 255),
}


def draw_hand(frame, pts, color):
    for c in HandLandmarksConnections.HAND_CONNECTIONS:
        a, b = pts[c.start], pts[c.end]
        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 200, 120), 2)
    for i, p in enumerate(pts):
        big = i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP)
        cv2.circle(frame, (int(p[0]), int(p[1])), 7 if big else 3,
                   color if big else (255, 255, 255), -1)


def draw_text(frame, lines, origin=(12, 30), scale=0.6, color=(255, 255, 255)):
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(28 * scale / 0.6)


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading hand model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    dry_run = "--dry-run" in sys.argv
    output = DryRunOutput() if (dry_run or sys.platform != "darwin") else MacOutput()
    ctl = Controller(output)
    hand_choice = HAND if HAND in HAND_CHOICES else "Any"
    swap = SWAP_HANDEDNESS

    ensure_model()
    landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,                       # see both, use the chosen one
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    ))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit("Could not open the camera. If macOS just asked for camera "
                 "permission, allow it and run again. Otherwise check System "
                 "Settings > Privacy & Security > Camera for your terminal app.")

    paused = False
    t0 = time.time()
    print(f"Output: {output.name}. Hand: {hand_choice}. Q quit, P pause, H cycle hand.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int((time.time() - t0) * 1000))

        m, pose, hand_note = None, "-", "no hand"
        chosen, _ = pick_hand(result, hand_choice, swap)
        for lms, label in label_hands(result, swap):
            pts = to_px(lms, w, h)
            if lms is chosen:
                m = measure(pts)
                pose = classify(m)
                hand_note = f"using {label}"
                draw_hand(frame, pts, STATE_COLOR[ctl.state])
                tag_color = STATE_COLOR[ctl.state]
            else:
                draw_hand(frame, pts, (90, 90, 90))          # ignored hand: grey
                tag_color = (120, 120, 120)
                if chosen is None:
                    hand_note = f"ignoring {label}"
            # label each hand near its wrist so the mapping is visible
            wx, wy = pts[WRIST]
            draw_text(frame, [label.upper()], origin=(int(wx) - 30, int(wy) + 28),
                      scale=0.7, color=tag_color)

        if paused:
            if ctl.state != "IDLE":
                ctl._go("IDLE", "idle (paused)")
        else:
            ctl.update(m)

        title = "PAUSED" if paused else ctl.state
        draw_text(frame, [title], origin=(12, 44), scale=1.2,
                  color=(0, 0, 255) if paused else STATE_COLOR[ctl.state])
        info = [f"hand: {hand_choice}  ({hand_note})   swap: {'on' if swap else 'off'}",
                f"pose: {pose}", f"last: {ctl.last_action}"]
        if m:
            info += [
                f"ext I{int(m['idx_ext'])} M{int(m['mid_ext'])} R{int(m['ring_ext'])} P{int(m['pink_ext'])}",
                f"idx-mid {m['d_im']:.2f}  thumb-idx {m['d_ti']:.2f}  thumb-knuckle {m['d_tk']:.2f}",
                f"spread {m['spread']:.2f}  tip_y {m['tip_y']:.2f}",
            ]
        draw_text(frame, info, origin=(12, 80), scale=0.55)
        draw_text(frame, [f"{output.name}   Q quit   P pause   H hand   S swap labels"],
                  origin=(12, h - 12), scale=0.5)

        cv2.imshow("Hand Puppet", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("p"), ord("P")):
            paused = not paused
        if key in (ord("h"), ord("H")):
            hand_choice = HAND_CHOICES[(HAND_CHOICES.index(hand_choice) + 1) % len(HAND_CHOICES)]
            ctl._go("IDLE", f"hand -> {hand_choice}")
        if key in (ord("s"), ord("S")):
            swap = not swap
            ctl._go("IDLE", f"swap labels -> {'on' if swap else 'off'}  "
                            f"(set SWAP_HANDEDNESS = {swap} to keep)")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
