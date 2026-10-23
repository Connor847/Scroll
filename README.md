# Hand Puppet

Scroll and zoom any macOS app with hand gestures, using the webcam.

A Python script watches one hand through the camera, recognizes a small set
of finger poses, and turns them into real input events: pixel-level scroll
wheel movement and Cmd+= / Cmd+- key presses. Because the events are posted
at the system level, they work in whatever app is in front: Safari, Chrome,
Preview, PDF readers, editors.

## Requirements

- macOS on Apple Silicon (the pinned `mediapipe` wheel is arm64 only; Intel
  Macs can use `mediapipe==0.10.21` with Python 3.9–3.12)
- Python 3.9 or newer
- A webcam
- Camera and Accessibility permission for the terminal app you run it from

## Install

```
git clone https://github.com/Connor847/Scroll.git
cd Scroll
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```
.venv/bin/python hand_control.py
```

Add `--dry-run` to print the actions in the terminal instead of sending them
to macOS. Useful for checking recognition before letting it drive.

On first run:

1. The hand-tracking model (`hand_landmarker.task`, about 8 MB) is downloaded
   next to the script.
2. macOS asks for **Camera** access for your terminal app. The run that
   triggers the prompt fails; allow it, quit the terminal app fully (Cmd+Q),
   reopen, run again.
3. The first time an event is sent, macOS asks for **Accessibility** access.
   Same procedure: allow, quit and reopen the terminal app, run again.

A preview window shows the camera, the tracked hand, the current state, and
the numbers the decisions are based on.

Keys in the preview window:

| Key | Action |
|---|---|
| Q | quit |
| P | pause / resume (no events are sent while paused) |
| H | cycle which hand is used: Right → Left → Any |
| S | swap the Left/Right labels if the tracker has them backwards |

## Gestures

Only the chosen hand is used (default: right). The other hand is ignored and
drawn in grey.

### Scroll

1. **Engage:** index and middle fingers extended and held together; ring and
   pinky curled; thumb relaxed, away from the fingertips.
2. **Move the two fingers up or down.** The page follows the movement, in the
   natural-scrolling direction (fingers up moves the content up). Stop moving
   and the page stops.
3. **Exit:** peace sign, open hand, or take the hand out of view.

### Zoom

1. **Arm:** peace sign (index and middle in a V, thumb tucked, ring and pinky
   curled). This opens a one-second window.
2. **Choose a direction** within the window:
   - **pinch** thumb, index and middle tips together → zoom **in**
   - **spread** thumb, index and middle wide apart → zoom **out**
3. **Zoom in:** spread the fingers; each step of spread sends one Cmd+=.
   Ends when the fingers are approximately fully extended.
   **Zoom out:** close the fingers; each step sends one Cmd+-. Ends when they
   are approximately fully pinched.
   Opening the hand also ends either.
4. Back to idle. The next zoom needs a new peace sign.

Each zoom is one-directional, set by the arming pose, and ratcheted: closing
slightly during a zoom-in does nothing. Zoom steps are keyboard shortcuts, so
apps zoom in their own increments (roughly 10% per step in most browsers).

### State diagram

```
IDLE ──two fingers together──▶ SCROLL ──peace──▶ ARMED
  │                               │
  └────────peace──────────────────┘
ARMED ──pinch──▶ ZOOM_IN  ──fully extended / open hand──▶ IDLE
ARMED ──spread─▶ ZOOM_OUT ──fully pinched  / open hand──▶ IDLE
ARMED ──1 s with neither──▶ IDLE
any state ──hand lost──▶ IDLE
```

## How it works

- **Tracking:** MediaPipe's HandLandmarker returns 21 landmarks per hand.
- **Pose recognition:** a finger counts as extended when its tip is farther
  from the wrist than its middle knuckle. Distances between fingertips are
  divided by the palm length (wrist to middle-finger knuckle), so the
  thresholds don't depend on how far the hand is from the camera.
- **Control signals:** scroll uses the frame-to-frame change in fingertip
  height, low-pass filtered; zoom uses the average distance from the thumb
  tip to the index and middle tips.
- **Stability:** a pose has to hold for a few consecutive frames before it
  changes state, and an engaged mode stays locked until an explicit release,
  so the hand can pass through other shapes mid-gesture.
- **Output:** Quartz `CGEventPost` with pixel-unit scroll wheel events and
  Cmd-modified key events. macOS has no public API for synthesizing trackpad
  pinch gestures, which is why zoom is stepped rather than continuous.

## Tuning

Every knob is at the top of `hand_control.py`. The preview shows the live
measurements (`idx-mid`, `thumb-idx`, `thumb-knuckle`, `spread`), so a
threshold can be set by holding the pose and reading the number.

| Knob | Default | Effect |
|---|---|---|
| `HAND` | `"Right"` | which hand to use; `"Left"` or `"Any"` |
| `SWAP_HANDEDNESS` | `True` | flip if the preview labels your hands backwards |
| `SCROLL_GAIN` | 700 | scroll pixels per palm-length of finger movement |
| `SCROLL_SMOOTHING` | 0.5 | 0 = raw, higher = smoother but laggier |
| `SCROLL_DEADZONE` | 0.008 | per-frame movement ignored as jitter |
| `SCROLL_INVERT` | `False` | flip scroll direction |
| `ZOOM_ARM_TIMEOUT` | 1.0 | seconds after the peace sign to pinch or spread |
| `ZOOM_STEP` | 0.15 | spread change per zoom step (lower = more steps) |
| `ZOOM_DONE_SPREAD` / `ZOOM_DONE_PINCH` | 1.30 / 0.35 | spread values that end zoom in / out |
| `HOLD_FRAMES` / `ARM_HOLD_FRAMES` | 3 / 2 | frames a pose must hold to change state |
| `PINCH_RADIUS`, `TOGETHER_MAX`, `APART_MIN`, `THUMB_OUT_MIN`, `THUMB_TUCKED_MAX`, `SCROLL_THUMB_CLEAR` | see file | pose recognition thresholds |

`THUMB_TUCKED_MAX` must stay below `THUMB_OUT_MIN`; the gap between them is
what separates a peace sign from the spread pose.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Could not open the camera` on the first run | The permission prompt just appeared. Allow it, Cmd+Q the terminal app, reopen, run again. |
| Still can't open the camera | The entry in System Settings → Privacy & Security → Camera must be for the app you're typing in (Terminal, iTerm2, Warp, VS Code...). `tccutil reset Camera <bundle-id>` forces a fresh prompt. |
| Recognizes poses but nothing happens in apps | Accessibility permission. System Settings → Privacy & Security → Accessibility, enable the terminal app, restart it. |
| It tracks the wrong hand | Press S to swap the labels (then set `SWAP_HANDEDNESS` to match), or H to change which hand is used. |
| Page scrolls the wrong way | `SCROLL_INVERT = True`. |
| Zoom-in never ends on its own | Your fully-extended `spread` reading is below `ZOOM_DONE_SPREAD`; lower it. Opening the hand always ends it. |
| Arm window keeps expiring | Raise `ZOOM_ARM_TIMEOUT`, or loosen `PINCH_RADIUS` / `THUMB_OUT_MIN`. The `last:` line in the preview says why a state ended. |

## Repository contents

| File | Purpose |
|---|---|
| `hand_control.py` | the app |
| `requirements.txt` | pinned dependencies |
| `index.html`, `download-mediapipe.sh` | an earlier in-browser prototype that draws strings between fingertips; not needed for the app. Run the script once to fetch its MediaPipe files, then serve the folder with `python3 -m http.server`. |

Not committed (see `.gitignore`): `.venv/`, the downloaded model, and the
browser prototype's MediaPipe files.
