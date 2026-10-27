#!/usr/bin/env python3
"""
Hand Puppet — Stage 3: scroll and zoom control, one chosen hand.

The webcam tracks the chosen hand (HAND setting, or press H to cycle).

  SCROLL  engage: index + middle extended and together, ring + pinky curled,
                  thumb clear of the fingertips.
          then:   move the fingers up/down; the page follows the movement
                  (natural direction, like a trackpad). Stop = stop.
          exit:   peace sign, open hand, or hand out of view.

  ZOOM    arm:    pointer (index finger up, other fingers curled). This opens
                  a short window (ZOOM_ARM_TIMEOUT).
          choose: within the window, either
                    pinch     thumb + index + middle tips together,
                              ring + pinky curled                  -> ZOOM IN
                    open hand all fingers extended                 -> ZOOM OUT
          then:   ZOOM IN:  spread the fingers; one Cmd+= per ZOOM_STEP of
                            new spread. Ends when approximately fully
                            extended (spread >= ZOOM_DONE_SPREAD), or on a
                            peace sign / open hand.
                  ZOOM OUT: close the whole hand; one Cmd+- per ZOOM_STEP of
                            new closure (hand "openness"). Ends when all four
                            fingers are curled. Peace sign and open hand are
                            ignored here because the closing motion passes
                            through both shapes.
                  Losing the hand ends either.
          after:  idle, until the next pointer.

  DISARM  peace sign (index + middle in a V, thumb tucked, ring + pinky
          curled) returns to idle from scroll, armed, or zoom-in.

Run:   python3 hand_control.py [--preview] [--dry-run]
       default    a small always-on-top status overlay at the top-left of the
                  screen shows the current mode. Click it to pause/resume.
                  Ctrl+C in the terminal quits.
       --preview  the full camera window instead, with the tracking drawn and
                  the live measurements, for tuning. Keys there:
                  Q quit   P pause/resume   H cycle hand   S swap L/R labels
       --dry-run  print the actions instead of sending them to macOS.
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
HOLD_FRAMES     = 2    # frames a pose must persist before it counts
                       # (raise to 3 if things engage by accident)
ARM_HOLD_FRAMES = 1    # same, for pinch / open hand during the zoom arm window

# -- scroll --
SCROLL_GAIN      = 550   # scroll pixels per palm-unit of finger movement
SCROLL_DEADZONE  = 0.008 # per-frame movement below this is ignored (jitter)
SCROLL_SMOOTHING = 0.5   # 0 = raw; higher = smoother but laggier (max ~0.8)
SCROLL_PAUSE_APART = 0.45  # while scrolling, if index & middle tips separate
                           # beyond this the scroll output pauses at once
                           # (before the release pose is confirmed), so
                           # opening the fingers to stop does not lurch
SCROLL_PAUSE_SEP_RATE = 0.06  # ...or if they separate faster than this per
                              # frame (catches the first frame of an exit)
SCROLL_INVERT    = False # set True if the page moves the wrong way

# -- zoom --
ZOOM_ARM_TIMEOUT = 1.0   # seconds after the peace sign to form pinch/spread
ZOOM_STEP        = 0.15  # ZOOM IN: palm-units of new spread per Cmd+=
ZOOM_OUT_STEPS   = 10    # ZOOM OUT: Cmd+- presses over one full hand close
ZOOM_OUT_CURVE   = 0.5   # how the steps are spread over the close: 1.0 =
                         # evenly over the openness measure (which barely
                         # moves early in the motion, then drops fast);
                         # lower = more of the steps happen early. 0.5 puts
                         # ~45% of the steps in the first 20% of closure.
ZOOM_OUT_DEADZONE = 0.15 # ZOOM OUT: closure below this (from the open
                         # position) counts as nothing; stops hand-shape
                         # jitter at the start from firing steps
ZOOM_OUT_REOPEN  = 0.25  # ZOOM OUT ends when openness rises this far above
                         # the lowest point reached (hand reopening, e.g. a
                         # peace sign after a close)
ZOOM_OUT_TIMEOUT = 1.5   # ZOOM OUT ends after this many seconds with no step
ZOOM_DONE_SPREAD = 1.30  # ZOOM IN ends when spread reaches this (fully extended)
ZOOM_DONE_OPENNESS = 1.05  # ZOOM OUT ends when hand openness drops to this
ZOOM_OUT_FIST_OPENNESS = 1.25  # ...or earlier if every finger reads curled
                               # and openness is already below this
ZOOM_INVERT      = False # set True if the directions come out backwards

# -- camera --
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 640    # smaller frames = faster tracking; 640x480 is plenty
CAMERA_HEIGHT = 480

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
        # how open the hand is: mean index + middle fingertip distance from
        # the wrist (~1.9 open, ~1.0 fist). Index + middle only, so that a
        # peace sign (which extends them) never reads as further closing.
        "openness": (dist(pts[INDEX_TIP], pts[WRIST]) +
                     dist(pts[MIDDLE_TIP], pts[WRIST])) / 2 / palm,
    }


def classify(m):
    """Pose name from a measurement dict:
    OPEN, PINCH, POINT, SPREAD, PEACE, SCROLL, or NEUTRAL."""
    if m["ring_ext"] or m["pink_ext"]:
        return "OPEN"
    if m["d_ti"] < PINCH_RADIUS and m["d_tm"] < PINCH_RADIUS and m["d_im"] < PINCH_RADIUS:
        return "PINCH"
    if m["idx_ext"] and not m["mid_ext"]:
        return "POINT"               # index up alone (thumb not considered)
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


RELEASE_POSES = ("OPEN", "PEACE")


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
# States:  IDLE -> SCROLL  (via SCROLL pose)  -> IDLE (PEACE/OPEN/lost)
#          IDLE -> ARMED   (via POINT)        -> IDLE (timeout/PEACE/OPEN/lost)
#          ARMED -> ZOOM_IN  (via PINCH) -> IDLE when fully extended / PEACE / OPEN / lost
#          ARMED -> ZOOM_OUT (via OPEN)  -> IDLE when hand closed to a fist / lost
# ---------------------------------------------------------------------------
class Controller:
    def __init__(self, output, clock=time.time):
        self.out = output
        self.clock = clock
        self.state = "IDLE"
        self.candidate, self.count = None, 0   # pose being counted toward HOLD_FRAMES
        self.armed_at = 0.0
        self.prev_y = None
        self.prev_d_im = None     # last index-middle gap, for separation rate
        self.smooth_dy = 0.0      # low-pass filtered scroll movement
        self.zoom_mark = None     # zoom-in: high-water spread
        self.zoom_start = None    # zoom-out: openness when the close began
        self.zoom_min = None      # zoom-out: lowest openness reached (ratchet)
        self.zoom_fired = 0       # zoom-out: steps sent so far
        self.zoom_last_t = 0.0    # zoom-out: time of engage / last step
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
        if self._held(pose, ("SCROLL", "POINT")):
            if pose == "SCROLL":
                self.prev_y = m["tip_y"]
                self.prev_d_im = m["d_im"]
                self.smooth_dy = 0.0
                self._go("SCROLL", "scroll engaged")
            else:
                self.armed_at = self.clock()
                self._go("ARMED", "armed: pinch = zoom in, spread = zoom out")

    def _in_scroll(self, m, pose):
        if self._held(pose, RELEASE_POSES):
            self._go("IDLE", f"idle ({pose.lower()})")
            return
        # Fingers separating (too far apart, or moving apart quickly) or a
        # finger curled: stop moving the page immediately and wait to see
        # whether this becomes a release.
        sep_rate = m["d_im"] - (self.prev_d_im if self.prev_d_im is not None else m["d_im"])
        self.prev_d_im = m["d_im"]
        if (pose in RELEASE_POSES
                or m["d_im"] > SCROLL_PAUSE_APART or sep_rate > SCROLL_PAUSE_SEP_RATE
                or not (m["idx_ext"] and m["mid_ext"])):
            self.prev_y = m["tip_y"]
            self.smooth_dy = 0.0
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
        if self._held(pose, ("PINCH", "OPEN", "PEACE"), ARM_HOLD_FRAMES):
            if pose == "PINCH":
                self.zoom_mark = m["spread"]
                self._go("ZOOM_IN", "zoom in: spread the fingers")
            elif pose == "OPEN":
                self.zoom_start = self.zoom_min = m["openness"]
                self.zoom_fired = 0
                self.zoom_last_t = self.clock()
                self._go("ZOOM_OUT", "zoom out: close the hand")
            else:
                self._go("IDLE", "idle (peace)")

    def _in_zoom_in(self, m, pose):
        if pose in RELEASE_POSES:
            self._go("IDLE", f"idle ({pose.lower()})")
            return
        if m["spread"] >= ZOOM_DONE_SPREAD:
            self._go("IDLE", "idle (zoom in finished)")
            return
        # ratchet: fire a step for every ZOOM_STEP of NEW spread beyond the
        # high-water mark; closing a little does nothing.
        while m["spread"] - self.zoom_mark >= ZOOM_STEP:
            self.zoom_mark += ZOOM_STEP
            self._zoom(+1)

    def _in_zoom_out(self, m, pose):
        # No pose-based exit here: closing a hand passes through the peace
        # and open shapes. Only a closed fist (or losing the hand) ends it.
        fist = not (m["idx_ext"] or m["mid_ext"] or m["ring_ext"] or m["pink_ext"])
        o = m["openness"]
        # If the hand was still opening when it was recognized, let the start
        # point follow it up; progress is measured from the widest point.
        if o > self.zoom_start and self.zoom_fired == 0:
            self.zoom_start = self.zoom_min = o
        self.zoom_min = min(self.zoom_min, o)               # ratchet
        closure = self.zoom_start - self.zoom_min - ZOOM_OUT_DEADZONE
        span = max(self.zoom_start - ZOOM_DONE_OPENNESS - ZOOM_OUT_DEADZONE, 0.3)
        progress = min(max(closure / span, 0.0), 1.0)
        progress = progress ** ZOOM_OUT_CURVE                # front-load
        target = int(progress * ZOOM_OUT_STEPS + 1e-9)
        while self.zoom_fired < target:
            self.zoom_fired += 1
            self.zoom_last_t = self.clock()
            self._zoom(-1)
        if o <= ZOOM_DONE_OPENNESS or (fist and o <= ZOOM_OUT_FIST_OPENNESS):
            self._go("IDLE", "idle (zoom out finished)")
        elif o >= self.zoom_min + ZOOM_OUT_REOPEN:
            self._go("IDLE", "idle (zoom out: hand reopened)")
        elif self.clock() - self.zoom_last_t > ZOOM_OUT_TIMEOUT:
            self._go("IDLE", "idle (zoom out: no movement)")

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
# Tracker: camera + model + controller, one frame at a time.
# Used by both display modes.
# ---------------------------------------------------------------------------
class Tracker:
    def __init__(self, output, hand_choice, swap):
        self.ctl = Controller(output)
        self.hand_choice = hand_choice
        self.swap = swap
        self.paused = False
        self.fps = 0.0
        self.size = (0, 0)
        self.hand_note = "no hand"
        self.pose = "-"
        self.m = None
        self.hands = []             # [(pts, label, is_chosen)] for drawing
        self._t0 = time.time()
        self._last = self._t0

        ensure_model()
        self.landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=2,                       # see both, use the chosen one
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        ))
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # always take the newest frame
        if not self.cap.isOpened():
            sys.exit("Could not open the camera. If macOS just asked for camera "
                     "permission, allow it and run again. Otherwise check System "
                     "Settings > Privacy & Security > Camera for your terminal app.")

    @property
    def state(self):
        return "PAUSED" if self.paused else self.ctl.state

    def step(self):
        """Grab one frame, run tracking + control. Returns the (mirrored)
        frame, or None if the camera stopped."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self._last, 1e-3))
        self._last = now
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        self.size = (w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int((now - self._t0) * 1000))

        self.m, self.pose, self.hand_note, self.hands = None, "-", "no hand", []
        chosen, _ = pick_hand(result, self.hand_choice, self.swap)
        for lms, label in label_hands(result, self.swap):
            pts = to_px(lms, w, h)
            if lms is chosen:
                self.m = measure(pts)
                self.pose = classify(self.m)
                self.hand_note = f"using {label}"
            elif chosen is None:
                self.hand_note = f"ignoring {label}"
            self.hands.append((pts, label, lms is chosen))

        if self.paused:
            if self.ctl.state != "IDLE":
                self.ctl._go("IDLE", "idle (paused)")
        else:
            self.ctl.update(self.m)
        return frame

    def close(self):
        self.cap.release()
        self.landmarker.close()


# ---------------------------------------------------------------------------
# Display mode 1: full camera preview window (--preview). For tuning.
# ---------------------------------------------------------------------------
def run_preview(tr, output):
    print(f"Output: {output.name}. Hand: {tr.hand_choice}. Q quit, P pause, H cycle hand, S swap.")
    while True:
        frame = tr.step()
        if frame is None:
            break
        w, h = tr.size
        ctl = tr.ctl
        for pts, label, chosen in tr.hands:
            color = STATE_COLOR[ctl.state] if chosen else (90, 90, 90)
            draw_hand(frame, pts, color)
            wx, wy = pts[WRIST]
            draw_text(frame, [label.upper()], origin=(int(wx) - 30, int(wy) + 28),
                      scale=0.7, color=color if chosen else (120, 120, 120))

        draw_text(frame, [tr.state], origin=(12, 44), scale=1.2,
                  color=(0, 0, 255) if tr.paused else STATE_COLOR[ctl.state])
        info = [f"hand: {tr.hand_choice}  ({tr.hand_note})   swap: {'on' if tr.swap else 'off'}",
                f"pose: {tr.pose}", f"last: {ctl.last_action}"]
        m = tr.m
        if m:
            info += [
                f"ext I{int(m['idx_ext'])} M{int(m['mid_ext'])} R{int(m['ring_ext'])} P{int(m['pink_ext'])}",
                f"idx-mid {m['d_im']:.2f}  thumb-idx {m['d_ti']:.2f}  thumb-knuckle {m['d_tk']:.2f}",
                f"spread {m['spread']:.2f}  openness {m['openness']:.2f}  tip_y {m['tip_y']:.2f}",
            ]
        draw_text(frame, info, origin=(12, 80), scale=0.55)
        draw_text(frame, [f"{output.name}   {tr.fps:.0f} fps   {w}x{h}   Q quit   P pause   H hand   S swap labels"],
                  origin=(12, h - 12), scale=0.5)

        cv2.imshow("Hand Puppet", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("p"), ord("P")):
            tr.paused = not tr.paused
        if key in (ord("h"), ord("H")):
            tr.hand_choice = HAND_CHOICES[(HAND_CHOICES.index(tr.hand_choice) + 1) % len(HAND_CHOICES)]
            ctl._go("IDLE", f"hand -> {tr.hand_choice}")
        if key in (ord("s"), ord("S")):
            tr.swap = not tr.swap
            ctl._go("IDLE", f"swap labels -> {'on' if tr.swap else 'off'}  "
                            f"(set SWAP_HANDEDNESS = {tr.swap} to keep)")
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Display mode 2 (default): small always-on-top status overlay, top-left.
#
# macOS wants the window on the main thread, so tracking runs in a background
# thread and a 20 Hz timer copies its state into the label.
# Click the overlay to pause/resume. Ctrl+C in the terminal quits.
# ---------------------------------------------------------------------------
OVERLAY_W, OVERLAY_H = 180, 56
OVERLAY_MARGIN = 8            # px from the screen's top-left corner
OVERLAY_ALPHA = 0.78          # background opacity

# Overlay colors as (r, g, b) 0..1
OVERLAY_COLOR = {
    "IDLE": (0.63, 0.63, 0.63), "SCROLL": (0.0, 0.9, 1.0), "ARMED": (1.0, 0.78, 0.0),
    "ZOOM_IN": (0.3, 0.86, 0.3), "ZOOM_OUT": (1.0, 0.47, 0.3), "PAUSED": (1.0, 0.25, 0.25),
}
OVERLAY_TEXT = {"ZOOM_IN": "ZOOM IN", "ZOOM_OUT": "ZOOM OUT"}


def run_overlay(tr, output):
    import signal
    import threading
    import objc
    from AppKit import (NSApplication, NSWindow, NSView, NSTextField, NSColor,
                        NSFont, NSScreen)
    from Foundation import NSObject, NSTimer, NSMakeRect

    stop = threading.Event()
    err = {"msg": None}

    def worker():
        try:
            while not stop.is_set():
                if tr.step() is None:
                    err["msg"] = "camera stopped"
                    break
        except Exception as e:          # surface tracker errors in the overlay
            err["msg"] = f"{type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()

    class ClickView(NSView):
        def acceptsFirstMouse_(self, event):
            return True

        def mouseDown_(self, event):
            tr.paused = not tr.paused

    class OverlayApp(NSObject):
        def tick_(self, timer):
            if stop.is_set():
                NSApplication.sharedApplication().terminate_(None)
                return
            state = tr.state
            text = OVERLAY_TEXT.get(state, state)
            if err["msg"]:
                text, state = "ERROR", "PAUSED"
            if text != self.last:
                self.last = text
                r, g, b = OVERLAY_COLOR.get(state, OVERLAY_COLOR["IDLE"])
                self.label.setStringValue_(text)
                self.label.setTextColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, 1.0))

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)        # accessory: no Dock icon, no menu bar takeover

    screen = NSScreen.mainScreen().visibleFrame()   # excludes the menu bar
    x = screen.origin.x + OVERLAY_MARGIN
    y = screen.origin.y + screen.size.height - OVERLAY_H - OVERLAY_MARGIN
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, y, OVERLAY_W, OVERLAY_H), 0, 2, False)   # borderless, buffered
    win.setLevel_(25)                                  # status-window level: above everything
    win.setCollectionBehavior_((1 << 0) | (1 << 4))    # all Spaces, stationary
    win.setOpaque_(False)
    win.setBackgroundColor_(NSColor.colorWithCalibratedWhite_alpha_(0.08, OVERLAY_ALPHA))
    win.setHasShadow_(True)
    win.setIgnoresMouseEvents_(False)

    view = ClickView.alloc().initWithFrame_(NSMakeRect(0, 0, OVERLAY_W, OVERLAY_H))
    win.setContentView_(view)
    label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 12, OVERLAY_W, 30))
    label.setBezeled_(False)
    label.setDrawsBackground_(False)
    label.setEditable_(False)
    label.setSelectable_(False)
    label.setAlignment_(1)                              # centered
    label.setFont_(NSFont.boldSystemFontOfSize_(22))
    label.setStringValue_("...")
    view.addSubview_(label)
    win.orderFrontRegardless()

    delegate = OverlayApp.alloc().init()
    delegate.label = label
    delegate.last = None
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        0.05, delegate, "tick:", None, True)

    def on_sigint(*_):
        stop.set()
    signal.signal(signal.SIGINT, on_sigint)

    threading.Thread(target=worker, daemon=True).start()
    print(f"Output: {output.name}. Hand: {tr.hand_choice}. Overlay at top-left; "
          f"click it to pause. Ctrl+C here to quit. (--preview for the camera window)")
    app.run()
    stop.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    dry_run = "--dry-run" in sys.argv
    preview = "--preview" in sys.argv or sys.platform != "darwin"
    output = DryRunOutput() if (dry_run or sys.platform != "darwin") else MacOutput()
    tr = Tracker(output, HAND if HAND in HAND_CHOICES else "Any", SWAP_HANDEDNESS)
    try:
        if preview:
            run_preview(tr, output)
        else:
            run_overlay(tr, output)
    finally:
        tr.close()


if __name__ == "__main__":
    main()
