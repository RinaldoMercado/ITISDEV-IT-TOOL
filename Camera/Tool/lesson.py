import os
import cv2
import time
import math
import random
import threading
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from gesture_matcher import (
    split_hands_by_handedness,
    extract_keypoints,
    load_action_templates,
    match_sequence,
    sequence_signature,
    sequence_profile,
    sequence_distance,
)

from ui_style import (
    load_style_vars,
    palette_from_vars,
    resolve_font_candidates,
    get_css_int,
    get_css_str,
)

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import winsound

    WINSOUND_AVAILABLE = True
except Exception:
    WINSOUND_AVAILABLE = False

# ManuMano Lesson: guided practice mode with real-time feedback and summary report.

# ----------------------------
# Visual Theme (ManuMano)
# ----------------------------
PALETTE_DEFAULT = {
    "brand_green": (57, 198, 74),
    "brand_green_soft": (126, 227, 126),
    "brand_green_dark": (31, 124, 49),
    "mint": (232, 243, 236),
    "card_white": (255, 255, 255),
    "card_gray": (230, 236, 232),
    "text_dark": (31, 31, 31),
    "text_muted": (106, 106, 106),
    "black": (0, 0, 0),
}

STYLE_VARS = load_style_vars()
PALETTE = palette_from_vars(STYLE_VARS, PALETTE_DEFAULT)


def rgb_to_bgr(color):
    return (color[2], color[1], color[0])


BRAND_GREEN_BGR = rgb_to_bgr(PALETTE["brand_green"])
BRAND_GREEN_SOFT_BGR = rgb_to_bgr(PALETTE["brand_green_soft"])
CARD_WHITE_BGR = rgb_to_bgr(PALETTE["card_white"])
CARD_GRAY_BGR = rgb_to_bgr(PALETTE["card_gray"])
BLACK_BGR = rgb_to_bgr(PALETTE["black"])

REGULAR_FONT_DEFAULTS = [
    os.path.join("fonts", "inter", "Inter-Regular.otf"),
    os.path.join("fonts", "inter", "Inter-Medium.otf"),
    os.path.join("fonts", "ambiguitytradition-regular.otf"),
    os.path.join("fonts", "Ambiguity-Regular.otf"),
    os.path.join("fonts", "Ambiguity-Regular.ttf"),
    os.path.join("assets", "fonts", "ambiguitytradition-regular.otf"),
    os.path.join("assets", "fonts", "Ambiguity-Regular.otf"),
    os.path.join("assets", "fonts", "Ambiguity-Regular.ttf"),
    "ambiguitytradition-regular.otf",
    "Ambiguity-Regular.otf",
    "Ambiguity-Regular.ttf",
    r"C:\Windows\Fonts\ambiguitytradition-regular.otf",
    r"C:\Windows\Fonts\Ambiguity-Regular.otf",
    r"C:\Windows\Fonts\Ambiguity-Regular.ttf",
]

BOLD_FONT_DEFAULTS = [
    os.path.join("fonts", "inter", "Inter-Bold.otf"),
    os.path.join("fonts", "inter", "Inter-SemiBold.otf"),
    os.path.join("fonts", "ambiguitytradition-bold.otf"),
    os.path.join("fonts", "Ambiguity-Bold.otf"),
    os.path.join("fonts", "Ambiguity-Bold.ttf"),
    os.path.join("assets", "fonts", "ambiguitytradition-bold.otf"),
    os.path.join("assets", "fonts", "Ambiguity-Bold.otf"),
    os.path.join("assets", "fonts", "Ambiguity-Bold.ttf"),
    "ambiguitytradition-bold.otf",
    "Ambiguity-Bold.otf",
    "Ambiguity-Bold.ttf",
    r"C:\Windows\Fonts\ambiguitytradition-bold.otf",
    r"C:\Windows\Fonts\Ambiguity-Bold.otf",
    r"C:\Windows\Fonts\Ambiguity-Bold.ttf",
]

REGULAR_FONT_CANDIDATES = resolve_font_candidates(STYLE_VARS, "font-regular", REGULAR_FONT_DEFAULTS)
BOLD_FONT_CANDIDATES = resolve_font_candidates(STYLE_VARS, "font-bold", BOLD_FONT_DEFAULTS)

REGULAR_FONT_PATH = next((p for p in REGULAR_FONT_CANDIDATES if os.path.exists(p)), None)
BOLD_FONT_PATH = next((p for p in BOLD_FONT_CANDIDATES if os.path.exists(p)), None)
if not BOLD_FONT_PATH:
    BOLD_FONT_PATH = REGULAR_FONT_PATH

FONT_CACHE = {}


def get_font(size, weight="regular"):
    if not PIL_AVAILABLE:
        return None

    path = BOLD_FONT_PATH if weight == "bold" else REGULAR_FONT_PATH
    if not path:
        return None

    size = max(12, int(size))
    cache_key = (weight, size)

    if cache_key not in FONT_CACHE:
        try:
            FONT_CACHE[cache_key] = ImageFont.truetype(path, size=size)
        except OSError:
            FONT_CACHE[cache_key] = ImageFont.load_default()

    return FONT_CACHE[cache_key]


def draw_panel(frame, x1, y1, x2, y2, color_bgr, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_texts(frame, items):
    if PIL_AVAILABLE and REGULAR_FONT_PATH:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_image)

        for item in items:
            text = item["text"]
            x, y = item["pos"]
            size = item.get("size", 24)
            color = item.get("color", PALETTE["text_dark"])
            align = item.get("align", "left")
            weight = item.get("weight", "regular")
            font = get_font(size, weight=weight)

            if align != "left":
                box = draw.textbbox((0, 0), text, font=font)
                width = box[2] - box[0]
                if align == "center":
                    x -= width // 2
                elif align == "right":
                    x -= width

            draw.text(
                (x, y),
                text,
                font=font,
                fill=color,
                stroke_width=1,
                stroke_fill=PALETTE["card_white"],
            )

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    for item in items:
        text = item["text"]
        x, y = item["pos"]
        size = item.get("size", 24)
        color_rgb = item.get("color", PALETTE["text_dark"])
        align = item.get("align", "left")

        scale = max(0.45, size / 36.0)
        thickness = max(1, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

        if align == "center":
            x -= tw // 2
        elif align == "right":
            x -= tw

        color_bgr = rgb_to_bgr(color_rgb)
        cv2.putText(frame, text, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, scale, CARD_WHITE_BGR, thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness, cv2.LINE_AA)

    return frame


def get_ui_scale(w, h):
    return max(0.72, min(1.9, min(w / 1280.0, h / 720.0)))


def sp(value, scale, minimum=1):
    return max(minimum, int(round(value * scale)))


def wrap_text(text, max_width, font_size):
    """Simple word wrap that uses PIL width if available."""
    words = text.split()
    if not words:
        return [""]

    if PIL_AVAILABLE and REGULAR_FONT_PATH:
        font = get_font(font_size, weight="regular")
        dummy = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(dummy)

        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            box = draw.textbbox((0, 0), test, font=font)
            width = box[2] - box[0]
            if width <= max_width or not current:
                current = test
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    # Fallback: approximate width by character count.
    approx_chars = max(10, int(max_width / (font_size * 0.55)))
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if len(test) <= approx_chars or not current:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def label_text(label):
    return label.replace("_", " ").title()


def format_countdown(seconds):
    """Format seconds as mm:ss for the timer pill."""
    total = max(0, int(seconds))
    minutes = total // 60
    secs = total % 60
    return f"{minutes:02d}:{secs:02d}"


def show_error_popup(title, message):
    """Best-effort popup for camera and runtime errors."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        print(f"{title}: {message}")


def open_camera(index=None, width=None, height=None):
    """Open the webcam with low-latency settings."""
    if index is None:
        index = CAMERA_INDEX
    if width is None:
        width = CAMERA_WIDTH
    if height is None:
        height = CAMERA_HEIGHT
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if cap is None or not cap.isOpened():
        show_error_popup(
            "Camera Unavailable",
            "ManuMano cannot access the camera. Close other apps using the camera and allow access in system settings.",
        )
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def play_sound_async(kind):
    """Non-blocking audio feedback for success/failure."""
    if not WINSOUND_AVAILABLE:
        return

    def _play():
        try:
            if kind == "success":
                winsound.MessageBeep(winsound.MB_OK)
            elif kind == "fail":
                winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()


def format_duration(seconds):
    if seconds is None:
        return "--:--"
    total = max(0, int(seconds))
    minutes = total // 60
    remainder = total % 60
    return f"{minutes:02d}:{remainder:02d}"


def point_in_rect(x, y, rect):
    return rect["x1"] <= x <= rect["x2"] and rect["y1"] <= y <= rect["y2"]


def fill_round_rect(frame, x1, y1, x2, y2, radius, color, border_color=None, border_thickness=1):
    """Draw a rounded rectangle with optional border."""
    radius = max(6, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)

    if border_color is not None and border_thickness > 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness, cv2.LINE_AA)


def compute_practice_layout(win_w, win_h, cam_w, cam_h):
    """Compute responsive layout slots to avoid stretching."""
    scale = get_ui_scale(win_w, win_h)
    header_h = max(sp(64, scale), int(win_h * 0.12))
    footer_h = max(sp(96, scale), int(win_h * 0.18))
    margin = sp(12, scale)

    preview_x1 = margin
    preview_y1 = header_h + margin
    preview_x2 = win_w - margin
    preview_y2 = win_h - footer_h - margin

    cam_aspect = cam_w / float(cam_h)
    bound_w = max(1, preview_x2 - preview_x1)
    bound_h = max(1, preview_y2 - preview_y1)
    bound_aspect = bound_w / float(bound_h)

    if bound_aspect > cam_aspect:
        new_h = bound_h
        new_w = int(new_h * cam_aspect)
        ox = preview_x1 + (bound_w - new_w) // 2
        oy = preview_y1
    else:
        new_w = bound_w
        new_h = int(new_w / cam_aspect)
        ox = preview_x1
        oy = preview_y1 + (bound_h - new_h) // 2

    return {
        "scale": scale,
        "header_h": header_h,
        "footer_h": footer_h,
        "margin": margin,
        "preview_bounds": (preview_x1, preview_y1, preview_x2, preview_y2),
        "preview_rect": (ox, oy, ox + new_w, oy + new_h),
        "footer_rect": (0, win_h - footer_h, win_w, win_h),
    }


def choose_target(actions, fail_counts):
    weights = [1.0 + (fail_counts[a] * 1.8) for a in actions]
    return random.choices(actions, weights=weights, k=1)[0]


# ---------------------------- PARAMETERS ----------------------------
POSE_MODEL = r"models/pose_landmarker.task"
HAND_MODEL = r"models/hand_landmarker.task"
LSTM_MODEL = r"action.h5"
DATA_PATH = "MP_Data"

SEQUENCE_LENGTH = 30
CONSISTENCY_FRAMES = 6
HAND_TIMEOUT = 1.5
PREDICT_EVERY = 1

ROUND_TIME_LIMIT = 9.0
FEEDBACK_TIME = 0.65
XP_GOAL = 8
XP_PER_SUCCESS = 1

HYBRID_ACCEPT_THRESHOLD = 0.56
MODEL_MIN_PROB = 0.14
TEMPLATE_MAX_DISTANCE = 0.48
MODEL_OVERRIDE_PROB = 0.70
TEMPLATE_OVERRIDE_MULT = 1.35
STABLE_REQUIRED_RATIO = 0.66
MAX_WRONG_PER_ROUND = 2

TARGET_MODEL_MIN_PROB = 0.10
TARGET_TEMPLATE_MAX_DISTANCE = 0.58
TARGET_COMBINED_MIN = 0.43
TARGET_STREAK_REQUIRED = 2
CANDIDATE_HOLD_ACCEPT_SECONDS = 0.8

# Pose/hand connections for overlay landmarks.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17),
    (12, 14), (14, 16), (16, 18),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Lesson review video settings.
SIGN_VIDEO_DIR = "FSL Videos"
SIGN_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
REVIEW_FPS_FALLBACK = 24.0
TARGET_FLASH_SECONDS = 0.8
FAIL_RED = (220, 76, 60)

# Camera settings tuned for lower latency.
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

ACTIONS = [
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
]

if not ACTIONS:
    print("No action folders found in MP_Data. Exiting.")
    raise SystemExit

# ---------------------------- GLOBALS ----------------------------
latest_pose_result = None
latest_hand_result = None
latest_handedness_result = None

sequence = []
predictions = []
latest_candidate = ""
latest_confidence = 0.0
latest_model_prob = 0.0
latest_template_distance = 999.0
target_pass_streak = 0
candidate_hold_label = ""
candidate_hold_start_time = None
last_attempt_label = ""

last_hand_seen_time = time.time()
frame_counter = 0

xp = 0
success_count = 0
wrong_xp_count = 0
fail_counts = {a: 0 for a in ACTIONS}
action_to_idx = {action: idx for idx, action in enumerate(ACTIONS)}

phase = "READY"  # READY -> ACTIVE -> FEEDBACK/REVIEW -> COMPLETE
current_target = choose_target(ACTIONS, fail_counts)
pending_target = None
round_start_time = 0.0
feedback_text = ""
feedback_until = 0.0
lesson_completed = False
wrong_attempts_in_round = 0
wrong_cooldown_until = 0.0
lesson_start_time = None
lesson_end_time = None
target_flash_until = 0.0
target_flash_color = None

# Review video state (shown after a wrong sign).
review_target = ""
review_video_cap = None
review_video_frame = None
review_last_frame_time = 0.0
review_video_fps = REVIEW_FPS_FALLBACK
review_next_clicked = False

ui_state = {
    "mouse_pos": (-1, -1),
    "click_pos": None,
}

window_state = {"fullscreen": False}
WINDOW_NAME = os.environ.get("MANUMANO_WINDOW_NAME", get_css_str(STYLE_VARS, "window-lesson", "ManuMano Lesson"))
KEEP_WINDOW_OPEN = os.environ.get("MANUMANO_KEEP_WINDOW") == "1"


def on_mouse(event, x, y, flags, param):
    ui_state["mouse_pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        ui_state["click_pos"] = (x, y)


def apply_window_mode(window_name, fullscreen):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

def window_closed(window_name):
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except Exception:
        return True



def get_window_size(window_name, fallback_w, fallback_h):
    try:
        _, _, w, h = cv2.getWindowImageRect(window_name)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return fallback_w, fallback_h


def compose_display_frame(frame, window_name, bg_color_bgr):
    src_h, src_w = frame.shape[:2]
    win_w, win_h = get_window_size(window_name, src_w, src_h)

    win_w = max(360, int(win_w))
    win_h = max(240, int(win_h))

    scale = min(win_w / float(src_w), win_h / float(src_h))
    stage_w = max(320, int(src_w * scale))
    stage_h = max(180, int(src_h * scale))
    stage_w = min(stage_w, win_w)
    stage_h = min(stage_h, win_h)

    stage = cv2.resize(frame, (stage_w, stage_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((win_h, win_w, 3), bg_color_bgr, dtype=np.uint8)
    ox = max(0, (win_w - stage_w) // 2)
    oy = max(0, (win_h - stage_h) // 2)

    return stage, canvas, ox, oy


def blit_stage(canvas, stage, ox, oy):
    sh, sw = stage.shape[:2]
    canvas[oy:oy + sh, ox:ox + sw] = stage
    return canvas


def draw_loading_screen(window_name, step, total_steps, title, subtitle):
    w, h = get_window_size(window_name, 1100, 680)
    w = max(780, int(w))
    h = max(460, int(h))

    frame = np.full((h, w, 3), rgb_to_bgr(PALETTE["mint"]), dtype=np.uint8)

    margin = 32
    card_x1 = margin
    card_y1 = margin
    card_x2 = w - margin
    card_y2 = h - margin

    draw_panel(frame, card_x1, card_y1, card_x2, card_y2, CARD_WHITE_BGR, 0.92)
    cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), CARD_GRAY_BGR, 2, cv2.LINE_AA)

    bar_margin = 90
    bar_y1 = int(h * 0.63)
    bar_y2 = bar_y1 + 22
    ratio = max(0.0, min(1.0, step / float(max(1, total_steps))))

    cv2.rectangle(frame, (bar_margin, bar_y1), (w - bar_margin, bar_y2), CARD_GRAY_BGR, -1)
    cv2.rectangle(frame, (bar_margin, bar_y1), (bar_margin + int((w - (2 * bar_margin)) * ratio), bar_y2), BRAND_GREEN_BGR, -1)

    items = [
        {
            "text": "ManuMano Lesson",
            "pos": (w // 2, int(h * 0.22)),
            "size": 52,
            "color": PALETTE["brand_green"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": title,
            "pos": (w // 2, int(h * 0.37)),
            "size": 30,
            "color": PALETTE["text_dark"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": subtitle,
            "pos": (w // 2, int(h * 0.45)),
            "size": 22,
            "color": PALETTE["text_muted"],
            "align": "center",
        },
        {
            "text": f"Loading {step}/{total_steps}",
            "pos": (w // 2, int(h * 0.70)),
            "size": 20,
            "color": PALETTE["brand_green_dark"],
            "align": "center",
            "weight": "bold",
        },
    ]

    frame = draw_texts(frame, items)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)

def target_template_distance(action, query_sig, query_profile, template_signatures, template_profiles):
    refs = template_signatures.get(action, None)
    profile_ref = template_profiles.get(action, None)

    if refs:
        point_dist = min(sequence_distance(query_sig, ref_sig) for ref_sig in refs)
    else:
        point_dist = 1.0

    profile_dist = 0.0
    if profile_ref is not None:
        denom = np.maximum(np.abs(profile_ref), 1e-4)
        profile_dist = float(np.mean(np.abs(query_profile - profile_ref) / denom))

    template_dist = point_dist + (0.35 * profile_dist)

    if profile_ref is not None:
        expects_contact = profile_ref[2] < 0.18
        expects_motion = profile_ref[0] > 0.18

        if expects_contact and query_profile[2] > profile_ref[2] + 0.08:
            template_dist += 0.08
        if expects_motion and query_profile[0] < profile_ref[0] * 0.55:
            template_dist += 0.08

    return float(template_dist)

# ---------------------------- CALLBACKS ----------------------------
def pose_callback(result, output_image, timestamp_ms):
    global latest_pose_result
    latest_pose_result = result.pose_landmarks


def hand_callback(result, output_image, timestamp_ms):
    global latest_hand_result, latest_handedness_result
    latest_hand_result = result.hand_landmarks
    latest_handedness_result = result.handedness


def draw_landmarks(frame, pose_landmarks, left_hand, right_hand, visibility_threshold=0.5):
    """Render pose and hand landmarks for practice feedback."""
    h, w, _ = frame.shape

    if pose_landmarks:
        for pose in pose_landmarks:
            for lm in pose:
                if getattr(lm, "visibility", 1.0) > visibility_threshold:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, BRAND_GREEN_SOFT_BGR, -1)
            for a, b in POSE_CONNECTIONS:
                p1, p2 = pose[a], pose[b]
                if (
                    getattr(p1, "visibility", 1.0) > visibility_threshold
                    and getattr(p2, "visibility", 1.0) > visibility_threshold
                ):
                    cv2.line(
                        frame,
                        (int(p1.x * w), int(p1.y * h)),
                        (int(p2.x * w), int(p2.y * h)),
                        BRAND_GREEN_BGR,
                        2,
                    )

    for hand in (left_hand, right_hand):
        if not hand:
            continue
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 128, 255), -1)
        for a, b in HAND_CONNECTIONS:
            p1, p2 = hand[a], hand[b]
            cv2.line(
                frame,
                (int(p1.x * w), int(p1.y * h)),
                (int(p2.x * w), int(p2.y * h)),
                (0, 128, 255),
                2,
            )


def reset_tracking_state():
    global latest_candidate, latest_confidence, latest_model_prob, latest_template_distance
    global target_pass_streak, candidate_hold_label, candidate_hold_start_time

    sequence.clear()
    predictions.clear()
    latest_candidate = ""
    latest_confidence = 0.0
    latest_model_prob = 0.0
    latest_template_distance = 999.0
    target_pass_streak = 0
    candidate_hold_label = ""
    candidate_hold_start_time = None


def get_sign_video_path(sign):
    """Resolve a local demo video for the given sign label."""
    if not sign:
        return None
    safe = sign.strip().lower().replace(" ", "_")
    folder_path = os.path.join(SIGN_VIDEO_DIR, safe)
    if os.path.isdir(folder_path):
        candidates = [
            os.path.join(folder_path, name)
            for name in os.listdir(folder_path)
            if name.lower().endswith(SIGN_VIDEO_EXTS)
        ]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    for ext in SIGN_VIDEO_EXTS:
        candidate = os.path.join(SIGN_VIDEO_DIR, f"{safe}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def open_review_video(sign):
    """Open the demo video for a sign, if available."""
    path = get_sign_video_path(sign)
    if not path:
        return None, REVIEW_FPS_FALLBACK

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = REVIEW_FPS_FALLBACK
    return cap, fps


def close_review_video():
    global review_video_cap, review_video_frame, review_last_frame_time

    if review_video_cap is not None:
        try:
            review_video_cap.release()
        except Exception:
            pass

    review_video_cap = None
    review_video_frame = None
    review_last_frame_time = 0.0


def update_review_video(now):
    """Advance the review video and return the latest frame."""
    global review_video_cap, review_video_frame, review_last_frame_time, review_video_fps

    if review_video_cap is None:
        return None

    if now - review_last_frame_time < (1.0 / max(1.0, review_video_fps)):
        return review_video_frame

    ok, frame = review_video_cap.read()
    if not ok:
        review_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = review_video_cap.read()

    if ok:
        review_video_frame = frame
        review_last_frame_time = now
    return review_video_frame


def draw_review_overlay(frame, target, video_frame, next_button):
    """Draw the wrong-sign review overlay with an optional demo video."""
    h, w, _ = frame.shape
    scale = get_ui_scale(w, h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), BLACK_BGR, -1)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    card_w = int(w * 0.76)
    card_h = int(h * 0.72)
    card_x1 = (w - card_w) // 2
    card_y1 = (h - card_h) // 2
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h

    draw_panel(frame, card_x1, card_y1, card_x2, card_y2, CARD_WHITE_BGR, 0.98)
    cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), CARD_GRAY_BGR, 2, cv2.LINE_AA)

    title = f"Correct sign: {label_text(target)}"
    frame = draw_texts(
        frame,
        [
            {
                "text": title,
                "pos": (w // 2, card_y1 + sp(18, scale)),
                "size": sp(26, scale, 16),
                "color": PALETTE["text_dark"],
                "align": "center",
                "weight": "bold",
            },
            {
                "text": "Watch the demo and tap Next to continue.",
                "pos": (w // 2, card_y1 + sp(52, scale)),
                "size": sp(16, scale, 11),
                "color": PALETTE["text_muted"],
                "align": "center",
            },
        ],
    )

    video_x1 = card_x1 + sp(28, scale)
    video_x2 = card_x2 - sp(28, scale)
    video_y1 = card_y1 + sp(80, scale)
    video_y2 = card_y2 - sp(84, scale)

    if video_frame is not None:
        vh, vw = video_frame.shape[:2]
        target_w = max(1, video_x2 - video_x1)
        target_h = max(1, video_y2 - video_y1)
        scale_fit = min(target_w / float(vw), target_h / float(vh))
        new_w = max(1, int(vw * scale_fit))
        new_h = max(1, int(vh * scale_fit))
        resized = cv2.resize(video_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        ox = video_x1 + (target_w - new_w) // 2
        oy = video_y1 + (target_h - new_h) // 2
        frame[oy:oy + new_h, ox:ox + new_w] = resized
        cv2.rectangle(frame, (video_x1, video_y1), (video_x2, video_y2), CARD_GRAY_BGR, 2, cv2.LINE_AA)
    else:
        draw_panel(frame, video_x1, video_y1, video_x2, video_y2, CARD_GRAY_BGR, 0.35)
        cv2.rectangle(frame, (video_x1, video_y1), (video_x2, video_y2), CARD_GRAY_BGR, 2, cv2.LINE_AA)
        frame = draw_texts(
            frame,
            [
                {
                    "text": "No demo video found.",
                    "pos": (w // 2, (video_y1 + video_y2) // 2 - sp(6, scale)),
                    "size": sp(18, scale, 12),
                    "color": PALETTE["text_dark"],
                    "align": "center",
                    "weight": "bold",
                },
                {
                    "text": f"Add {label_text(target)} video to {SIGN_VIDEO_DIR}.",
                    "pos": (w // 2, (video_y1 + video_y2) // 2 + sp(18, scale)),
                    "size": sp(14, scale, 11),
                    "color": PALETTE["text_muted"],
                    "align": "center",
                },
            ],
        )

    btn_w = int(card_w * 0.42)
    btn_h = sp(46, scale, 32)
    btn_x1 = w // 2 - btn_w // 2
    btn_y1 = card_y2 - sp(58, scale)
    btn_x2 = btn_x1 + btn_w
    btn_y2 = btn_y1 + btn_h
    next_button.update({"x1": btn_x1, "y1": btn_y1, "x2": btn_x2, "y2": btn_y2})

    fill_round_rect(frame, btn_x1, btn_y1, btn_x2, btn_y2, sp(16, scale, 10), BRAND_GREEN_BGR, None, 0)
    frame = draw_texts(
        frame,
        [
            {
                "text": "Next",
                "pos": (w // 2, btn_y1 + sp(10, scale)),
                "size": sp(20, scale, 14),
                "color": PALETTE["card_white"],
                "align": "center",
                "weight": "bold",
            }
        ],
    )

    return frame

# ---------------------------- UI ----------------------------
def draw_lesson_hud(
    canvas,
    layout,
    target,
    phase,
    time_left,
    feedback_text,
    feedback_visible,
    confidence,
    hands_visible,
    xp,
    xp_goal,
    target_color,
    last_attempt_label,
    candidate_label,
):
    """Draw ManuMano practice UI to match the mobile template."""
    h, w, _ = canvas.shape
    scale = layout["scale"]
    header_h = layout["header_h"]
    margin = layout["margin"]
    footer_x1, footer_y1, footer_x2, footer_y2 = layout["footer_rect"]
    preview_bounds = layout["preview_bounds"]
    preview_rect = layout["preview_rect"]

    # Header bar
    cv2.rectangle(canvas, (0, 0), (w, header_h), BRAND_GREEN_BGR, -1)

    # Progress line (prominent, sits between preview and footer).
    if xp_goal > 0:
        ratio = max(0.0, min(1.0, xp / float(xp_goal)))
        line_y = footer_y1 - sp(10, scale)
        line_x1 = margin
        line_x2 = w - margin
        cv2.rectangle(canvas, (line_x1, line_y), (line_x2, line_y + sp(6, scale)), CARD_GRAY_BGR, -1)
        cv2.rectangle(
            canvas,
            (line_x1, line_y),
            (line_x1 + int((line_x2 - line_x1) * ratio), line_y + sp(6, scale)),
            BRAND_GREEN_SOFT_BGR,
            -1,
        )

    # Preview border (keeps the camera area framed without stretching).
    pbx1, pby1, pbx2, pby2 = preview_bounds
    cv2.rectangle(canvas, (pbx1, pby1), (pbx2, pby2), CARD_GRAY_BGR, 1, cv2.LINE_AA)

    # Timer pill in the preview area.
    prx1, pry1, prx2, pry2 = preview_rect
    pill_w = sp(98, scale, 64)
    pill_h = sp(30, scale, 22)
    pill_x2 = prx2 - sp(10, scale)
    pill_x1 = max(prx1 + sp(10, scale), pill_x2 - pill_w)
    pill_y1 = pry1 + sp(10, scale)
    pill_y2 = pill_y1 + pill_h
    fill_round_rect(
        canvas,
        pill_x1,
        pill_y1,
        pill_x2,
        pill_y2,
        sp(12, scale, 8),
        CARD_WHITE_BGR,
        CARD_GRAY_BGR,
        1,
    )

    display_time = format_countdown(time_left if phase == "ACTIVE" else 0)
    lower_feedback = feedback_text.lower() if feedback_text else ""
    is_fail_feedback = "wrong" in lower_feedback or "time up" in lower_feedback or "review" in lower_feedback
    timer_color = FAIL_RED if is_fail_feedback else PALETTE["brand_green_dark"]

    # Target pill inside the preview.
    target_title = label_text(target)
    target_color = target_color if target_color is not None else PALETTE["text_dark"]
    available_w = max(sp(140, scale, 120), prx2 - prx1 - sp(32, scale))
    target_pill_w = max(sp(140, scale, 120), min(available_w, sp(280, scale, 150)))
    target_pill_h = sp(46, scale, 30)
    target_pill_x1 = (prx1 + prx2 - target_pill_w) // 2
    target_pill_y2 = pry2 - sp(14, scale)
    target_pill_y1 = target_pill_y2 - target_pill_h
    fill_round_rect(
        canvas,
        target_pill_x1,
        target_pill_y1,
        target_pill_x1 + target_pill_w,
        target_pill_y2,
        sp(18, scale, 10),
        CARD_WHITE_BGR,
        CARD_GRAY_BGR,
        1,
    )

    # Footer card.
    card_pad = sp(10, scale)
    card_x1 = footer_x1 + card_pad
    card_x2 = footer_x2 - card_pad
    card_y1 = footer_y1 + card_pad
    card_y2 = footer_y2 - card_pad
    shadow = sp(4, scale)
    draw_panel(canvas, card_x1 + shadow, card_y1 + shadow, card_x2 + shadow, card_y2 + shadow, BLACK_BGR, 0.08)
    fill_round_rect(canvas, card_x1, card_y1, card_x2, card_y2, sp(20, scale, 12), CARD_WHITE_BGR, CARD_GRAY_BGR, 1)

    match_percent = int(max(0.0, min(1.0, confidence)) * 100) if hands_visible else 0

    footer_items = [
        {
            "text": "Practice",
            "pos": (w // 2, sp(16, scale)),
            "size": sp(22, scale, 14),
            "color": PALETTE["card_white"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": display_time,
            "pos": (pill_x2 - sp(10, scale), pill_y1 + sp(4, scale)),
            "size": sp(16, scale, 11),
            "color": timer_color,
            "align": "right",
            "weight": "bold",
        },
        {
            "text": target_title,
            "pos": (target_pill_x1 + target_pill_w // 2, target_pill_y1 + sp(6, scale)),
            "size": sp(20, scale, 13),
            "color": target_color,
            "align": "center",
            "weight": "bold",
        },
    ]

    center_x = (card_x1 + card_x2) // 2
    line_y1 = card_y1 + sp(16, scale)
    line_y2 = card_y1 + sp(46, scale)

    if feedback_visible:
        feedback_color = FAIL_RED if is_fail_feedback else PALETTE["brand_green"]
        footer_items.append(
            {
                "text": feedback_text,
                "pos": (center_x, card_y1 + sp(30, scale)),
                "size": sp(24, scale, 14),
                "color": feedback_color,
                "align": "center",
                "weight": "bold",
            }
        )
    elif phase == "READY":
        footer_items.append(
            {
                "text": "Tap to start",
                "pos": (center_x, card_y1 + sp(30, scale)),
                "size": sp(24, scale, 14),
                "color": PALETTE["brand_green"],
                "align": "center",
                "weight": "bold",
            }
        )
    elif phase == "ACTIVE":
        footer_items.extend(
            [
                {
                    "text": "What you're signing:",
                    "pos": (center_x, line_y1),
                    "size": sp(16, scale, 11),
                    "color": PALETTE["text_muted"],
                    "align": "center",
                    "weight": "bold",
                },
                {
                    "text": candidate_label if candidate_label else "--",
                    "pos": (center_x, line_y2),
                    "size": sp(22, scale, 14),
                    "color": PALETTE["text_dark"],
                    "align": "center",
                    "weight": "bold",
                },
            ]
        )
    elif phase == "REVIEW":
        footer_items.extend(
            [
                {
                    "text": "What you signed:",
                    "pos": (center_x, line_y1),
                    "size": sp(16, scale, 11),
                    "color": PALETTE["text_muted"],
                    "align": "center",
                    "weight": "bold",
                },
                {
                    "text": label_text(last_attempt_label) if last_attempt_label else "--",
                    "pos": (center_x, line_y2),
                    "size": sp(22, scale, 14),
                    "color": FAIL_RED,
                    "align": "center",
                    "weight": "bold",
                },
            ]
        )
    else:
        footer_items.append(
            {
                "text": "Get ready for the next sign",
                "pos": (center_x, card_y1 + sp(30, scale)),
                "size": sp(18, scale, 12),
                "color": PALETTE["text_muted"],
                "align": "center",
            }
        )

    footer_items.append(
        {
            "text": f"Match: {match_percent}%" if hands_visible else "Match: --",
            "pos": (card_x1 + sp(16, scale), card_y2 - sp(22, scale)),
            "size": sp(14, scale, 10),
            "color": PALETTE["text_muted"] if not hands_visible else PALETTE["text_dark"],
            "weight": "bold",
        }
    )

    return draw_texts(canvas, footer_items)


def draw_summary_screen(frame, fail_counts, success_count, wrong_xp_count, xp_goal, elapsed_seconds, button):
    """Draw the end-of-lesson report UI."""
    h, w, _ = frame.shape
    scale = get_ui_scale(w, h)

    header_h = max(sp(64, scale), int(h * 0.12))
    cv2.rectangle(frame, (0, 0), (w, header_h), BRAND_GREEN_BGR, -1)

    card_margin = max(sp(18, scale), int(min(w, h) * 0.04))
    card_x1 = card_margin
    card_x2 = w - card_margin
    card_y1 = header_h + card_margin
    card_y2 = h - card_margin
    fill_round_rect(frame, card_x1, card_y1, card_x2, card_y2, sp(24, scale, 14), CARD_WHITE_BGR, CARD_GRAY_BGR, 1)

    failing = [(s, c) for s, c in fail_counts.items() if c > 0]
    failing.sort(key=lambda x: x[1], reverse=True)

    texts = [
        {
            "text": "Report",
            "pos": (w // 2, sp(16, scale)),
            "size": sp(22, scale, 14),
            "color": PALETTE["card_white"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": "Great Job!",
            "pos": (w // 2, card_y1 + sp(22, scale)),
            "size": sp(30, scale, 18),
            "color": PALETTE["brand_green"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": "Here's how you did:",
            "pos": (w // 2, card_y1 + sp(56, scale)),
            "size": sp(19, scale, 12),
            "color": PALETTE["text_dark"],
            "align": "center",
            "weight": "bold",
        },
    ]

    metric_y = card_y1 + sp(98, scale)
    metric_gap = sp(36, scale)
    label_x = card_x1 + sp(24, scale)
    value_x = card_x2 - sp(24, scale)

    texts.extend(
        [
            {
                "text": "Successes:",
                "pos": (label_x, metric_y),
                "size": sp(19, scale, 12),
                "color": PALETTE["text_dark"],
                "weight": "bold",
            },
            {
                "text": f"{success_count}",
                "pos": (value_x, metric_y),
                "size": sp(19, scale, 12),
                "color": PALETTE["brand_green"],
                "align": "right",
                "weight": "bold",
            },
            {
                "text": "Mistakes:",
                "pos": (label_x, metric_y + metric_gap),
                "size": sp(19, scale, 12),
                "color": PALETTE["text_dark"],
                "weight": "bold",
            },
            {
                "text": f"{wrong_xp_count}",
                "pos": (value_x, metric_y + metric_gap),
                "size": sp(19, scale, 12),
                "color": FAIL_RED,
                "align": "right",
                "weight": "bold",
            },
        ]
    )

    tip_signs = [label_text(sign) for sign, _ in failing[:3]]
    if tip_signs:
        modules = [f"Module {i + 1}" for i in range(len(tip_signs))]

        def join_list(items):
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            if len(items) == 2:
                return f"{items[0]} and {items[1]}"
            return ", ".join(items[:-1]) + f", and {items[-1]}"

        sign_list = join_list(tip_signs)
        module_list = join_list(modules)
        tip_text = (
            f"Tip: Your performance in signing {sign_list} can still be improved, "
            f"you can study more about them in {module_list}."
        )
    else:
        tip_text = "Tip: Your performance was strong across this lesson. Keep practicing to stay sharp."

    info_y1 = metric_y + sp(74, scale)
    info_x1 = card_x1 + sp(18, scale)
    info_x2 = card_x2 - sp(18, scale)
    info_color = (234, 244, 255)

    icon_r = sp(16, scale, 12)
    text_left = info_x1 + sp(48, scale)
    line_height = sp(18, scale, 12)
    tip_lines = wrap_text(tip_text, info_x2 - text_left - sp(12, scale), sp(16, scale, 11))
    tip_lines = tip_lines[:3] if tip_lines else [""]
    line_count = max(1, len(tip_lines))

    text_block_h = line_count * line_height
    content_h = max(icon_r * 2, text_block_h)
    info_h = content_h + sp(18, scale)
    max_info_y2 = card_y2 - sp(120, scale)
    info_y2 = info_y1 + info_h

    while info_y2 > max_info_y2 and line_count > 1:
        line_count -= 1
        text_block_h = line_count * line_height
        content_h = max(icon_r * 2, text_block_h)
        info_h = content_h + sp(18, scale)
        info_y2 = info_y1 + info_h

    if info_y2 > max_info_y2:
        info_y2 = max_info_y2

    fill_round_rect(frame, info_x1, info_y1, info_x2, info_y2, sp(16, scale, 10), rgb_to_bgr(info_color), None, 0)
    cv2.rectangle(frame, (info_x1, info_y1), (info_x2, info_y2), CARD_GRAY_BGR, 1, cv2.LINE_AA)

    content_y1 = info_y1 + (info_h - content_h) // 2
    icon_cx = info_x1 + sp(22, scale)
    icon_cy = content_y1 + (content_h // 2)
    cv2.circle(frame, (icon_cx, icon_cy), icon_r, BRAND_GREEN_BGR, -1, cv2.LINE_AA)
    frame = draw_texts(
        frame,
        [
            {
                "text": "i",
                "pos": (icon_cx, icon_cy - sp(10, scale)),
                "size": sp(20, scale, 12),
                "color": PALETTE["card_white"],
                "align": "center",
                "weight": "bold",
            }
        ],
    )

    line_y = content_y1 + (content_h - text_block_h) // 2
    for line in tip_lines[:line_count]:
        texts.append(
            {
                "text": line,
                "pos": (text_left, line_y),
                "size": sp(16, scale, 11),
                "color": PALETTE["text_dark"],
                "weight": "bold",
            }
        )
        line_y += line_height

    time_text = f"Time: {format_duration(elapsed_seconds)}"
    texts.append(
        {
            "text": time_text,
            "pos": (label_x, card_y2 - sp(88, scale)),
            "size": sp(16, scale, 11),
            "color": PALETTE["text_muted"],
            "weight": "bold",
        }
    )

    btn_w = int((card_x2 - card_x1) * 0.7)
    btn_h = sp(46, scale, 32)
    btn_x1 = (w - btn_w) // 2
    btn_y2 = card_y2 - sp(22, scale)
    btn_y1 = btn_y2 - btn_h
    button.update({"x1": btn_x1, "y1": btn_y1, "x2": btn_x1 + btn_w, "y2": btn_y2})

    fill_round_rect(frame, btn_x1, btn_y1, btn_x1 + btn_w, btn_y2, sp(18, scale, 10), BRAND_GREEN_BGR, None, 0)
    texts.append(
        {
            "text": "Continue",
            "pos": (w // 2, btn_y1 + sp(10, scale)),
            "size": sp(20, scale, 13),
            "color": PALETTE["card_white"],
            "align": "center",
            "weight": "bold",
        }
    )

    return draw_texts(frame, texts)


# ---------------------------- INIT ----------------------------
apply_window_mode(WINDOW_NAME, window_state["fullscreen"])
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

TOTAL_LOAD_STEPS = max(2, get_css_int(STYLE_VARS, "mode-loading-steps", 4))
draw_loading_screen(WINDOW_NAME, 1, TOTAL_LOAD_STEPS, "Preparing lesson", "Initializing pose tracker")
pose_landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=pose_callback,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
)

draw_loading_screen(WINDOW_NAME, 2, TOTAL_LOAD_STEPS, "Preparing lesson", "Initializing hand tracker")
hand_landmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=hand_callback,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
)

draw_loading_screen(WINDOW_NAME, 3, TOTAL_LOAD_STEPS, "Preparing lesson", "Loading LSTM model")
model = load_model(LSTM_MODEL, compile=False)

draw_loading_screen(WINDOW_NAME, 4, TOTAL_LOAD_STEPS, "Preparing lesson", "Loading sign templates")
template_signatures, template_profiles = load_action_templates(
    data_path=DATA_PATH,
    actions=ACTIONS,
    sequence_length=SEQUENCE_LENGTH,
    max_templates_per_action=3,
)

loaded_template_total = sum(len(v) for v in template_signatures.values())
print(f"Loaded {loaded_template_total} frame-match templates from MP_Data")

if REGULAR_FONT_PATH:
    print(f"Using regular font: {REGULAR_FONT_PATH}")
if BOLD_FONT_PATH:
    print(f"Using bold font: {BOLD_FONT_PATH}")

cap = open_camera()
if cap is None:
    pose_landmarker.close()
    hand_landmarker.close()
    raise SystemExit

print("\nLesson controls: SPACE starts each target, ESC or window X closes, F fullscreen.\n")
start_time = time.time()

# ---------------------------- LOOP ----------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        left_hand = None
        right_hand = None
        hands_visible = False

        if phase != "REVIEW":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int((now - start_time) * 1000)

            pose_landmarker.detect_async(mp_image, timestamp)
            hand_landmarker.detect_async(mp_image, timestamp)

            left_hand, right_hand = split_hands_by_handedness(latest_hand_result, latest_handedness_result)
            hands_visible = left_hand is not None or right_hand is not None

        if phase != "REVIEW":
            if hands_visible:
                last_hand_seen_time = now
            elif now - last_hand_seen_time > HAND_TIMEOUT:
                reset_tracking_state()

        if phase == "FEEDBACK" and now >= feedback_until:
            current_target = pending_target if pending_target else choose_target(ACTIONS, fail_counts)
            pending_target = None
            phase = "READY"
            wrong_attempts_in_round = 0
            round_start_time = 0.0
            reset_tracking_state()

        if phase == "ACTIVE":
            if hands_visible:
                keypoints = extract_keypoints(latest_pose_result, left_hand, right_hand)
                sequence.append(keypoints)
                sequence = sequence[-SEQUENCE_LENGTH:]

                if len(sequence) == SEQUENCE_LENGTH:
                    frame_counter += 1

                    if frame_counter % PREDICT_EVERY == 0:
                        model_probs = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        match = match_sequence(
                            sequence=sequence,
                            model_probs=model_probs,
                            actions=ACTIONS,
                            template_signatures=template_signatures,
                            template_profiles=template_profiles,
                        )

                        latest_candidate = match["action"]
                        latest_confidence = match["combined"]
                        latest_model_prob = match["model_prob"]
                        latest_template_distance = match["template_dist"]

                        if latest_candidate == current_target:
                            if candidate_hold_label != current_target:
                                candidate_hold_label = current_target
                                candidate_hold_start_time = now
                        else:
                            candidate_hold_label = ""
                            candidate_hold_start_time = None

                        predictions.append(latest_candidate)
                        predictions = predictions[-CONSISTENCY_FRAMES:]

                        stable_needed = max(3, int(math.ceil(CONSISTENCY_FRAMES * STABLE_REQUIRED_RATIO)))
                        stable = predictions.count(latest_candidate) >= stable_needed
                        accepted_match = (
                            (
                                latest_confidence > HYBRID_ACCEPT_THRESHOLD
                                and latest_model_prob > MODEL_MIN_PROB
                                and latest_template_distance < TEMPLATE_MAX_DISTANCE
                            )
                            or (
                                latest_model_prob > MODEL_OVERRIDE_PROB
                                and latest_template_distance < (TEMPLATE_MAX_DISTANCE * TEMPLATE_OVERRIDE_MULT)
                            )
                        )

                        target_idx = action_to_idx.get(current_target, -1)
                        target_model_prob = (
                            float(model_probs[target_idx])
                            if 0 <= target_idx < len(model_probs)
                            else 0.0
                        )

                        query_seq = np.asarray(sequence, dtype=np.float32)
                        query_sig = sequence_signature(query_seq)
                        query_profile = sequence_profile(query_seq)
                        target_dist = target_template_distance(
                            current_target,
                            query_sig,
                            query_profile,
                            template_signatures,
                            template_profiles,
                        )
                        target_sim = 1.0 / (1.0 + 6.0 * max(0.0, target_dist))
                        target_combined = (0.35 * target_model_prob) + (0.65 * target_sim)

                        target_frame_pass = (
                            target_dist < TARGET_TEMPLATE_MAX_DISTANCE
                            and (
                                target_model_prob > TARGET_MODEL_MIN_PROB
                                or target_combined > TARGET_COMBINED_MIN
                            )
                        )

                        if target_frame_pass:
                            target_pass_streak += 1
                        else:
                            target_pass_streak = max(0, target_pass_streak - 1)

                        candidate_hold_elapsed = (
                            now - candidate_hold_start_time
                            if candidate_hold_start_time is not None
                            else 0.0
                        )
                        hold_accept = (
                            latest_candidate == current_target
                            and candidate_hold_start_time is not None
                            and candidate_hold_elapsed >= CANDIDATE_HOLD_ACCEPT_SECONDS
                        )

                        awarded_correct = hold_accept or (target_pass_streak >= TARGET_STREAK_REQUIRED)
                        if (
                            not awarded_correct
                            and stable
                            and accepted_match
                            and latest_candidate == current_target
                        ):
                            awarded_correct = True

                        if awarded_correct:
                            xp += XP_PER_SUCCESS
                            success_count += 1
                            last_attempt_label = current_target
                            if lesson_start_time is None:
                                lesson_start_time = now
                            target_flash_color = PALETTE["brand_green"]
                            target_flash_until = now + TARGET_FLASH_SECONDS
                            play_sound_async("success")

                            feedback_text = "Great Job!" if max(latest_confidence, target_combined) > 0.82 else "Correct!"
                            feedback_until = now + FEEDBACK_TIME

                            latest_candidate = current_target
                            latest_confidence = max(latest_confidence, target_combined)
                            latest_model_prob = max(latest_model_prob, target_model_prob)
                            latest_template_distance = min(latest_template_distance, target_dist)

                            if xp >= XP_GOAL:
                                phase = "COMPLETE"
                                lesson_completed = True
                                lesson_end_time = now
                            else:
                                pending_target = choose_target(ACTIONS, fail_counts)
                                phase = "FEEDBACK"

                            wrong_attempts_in_round = 0
                            reset_tracking_state()

                        elif stable and accepted_match and latest_candidate != current_target and now >= wrong_cooldown_until:
                            wrong_cooldown_until = now + 0.45
                            wrong_attempts_in_round += 1

                            if wrong_attempts_in_round >= MAX_WRONG_PER_ROUND:
                                xp += XP_PER_SUCCESS
                                wrong_xp_count += 1
                                fail_counts[current_target] += 1
                                last_attempt_label = latest_candidate or current_target
                                feedback_text = "Let's review that sign"
                                feedback_until = now + FEEDBACK_TIME
                                target_flash_color = FAIL_RED
                                target_flash_until = now + TARGET_FLASH_SECONDS
                                play_sound_async("fail")

                                if xp >= XP_GOAL:
                                    phase = "COMPLETE"
                                    lesson_completed = True
                                    lesson_end_time = now
                                else:
                                    pending_target = choose_target(ACTIONS, fail_counts)
                                    phase = "REVIEW"
                                    review_target = current_target
                                    close_review_video()
                                    review_video_cap, review_video_fps = open_review_video(review_target)
                                    review_next_clicked = False

                                wrong_attempts_in_round = 0
                            else:
                                feedback_text = "Try again"
                                feedback_until = now + FEEDBACK_TIME
                                round_start_time = now

                            reset_tracking_state()

            time_left = ROUND_TIME_LIMIT - (now - round_start_time)
            if time_left <= 0 and phase == "ACTIVE":
                wrong_attempts_in_round += 1

                if wrong_attempts_in_round >= MAX_WRONG_PER_ROUND:
                    xp += XP_PER_SUCCESS
                    wrong_xp_count += 1
                    fail_counts[current_target] += 1
                    last_attempt_label = latest_candidate or current_target
                    feedback_text = "Let's review that sign"
                    feedback_until = now + FEEDBACK_TIME
                    target_flash_color = FAIL_RED
                    target_flash_until = now + TARGET_FLASH_SECONDS
                    play_sound_async("fail")

                    if xp >= XP_GOAL:
                        phase = "COMPLETE"
                        lesson_completed = True
                        lesson_end_time = now
                    else:
                        pending_target = choose_target(ACTIONS, fail_counts)
                        phase = "REVIEW"
                        review_target = current_target
                        close_review_video()
                        review_video_cap, review_video_fps = open_review_video(review_target)
                        review_next_clicked = False

                    wrong_attempts_in_round = 0
                else:
                    feedback_text = "Time's up"
                    feedback_until = now + FEEDBACK_TIME
                    round_start_time = now

                reset_tracking_state()
        else:
            time_left = ROUND_TIME_LIMIT

        feedback_visible = now < feedback_until

        win_w, win_h = get_window_size(WINDOW_NAME, frame.shape[1], frame.shape[0])
        win_w = max(420, int(win_w))
        win_h = max(300, int(win_h))

        layout = compute_practice_layout(win_w, win_h, frame.shape[1], frame.shape[0])
        canvas = np.full((win_h, win_w, 3), rgb_to_bgr(PALETTE["mint"]), dtype=np.uint8)

        preview_frame = frame.copy()
        if hands_visible:
            draw_landmarks(preview_frame, latest_pose_result, left_hand, right_hand)

        vx1, vy1, vx2, vy2 = layout["preview_rect"]
        preview_resized = cv2.resize(
            preview_frame,
            (max(1, vx2 - vx1), max(1, vy2 - vy1)),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas[vy1:vy2, vx1:vx2] = preview_resized

        target_color = target_flash_color if now < target_flash_until else None

        candidate_label = label_text(latest_candidate) if latest_candidate else "--"

        canvas = draw_lesson_hud(
            canvas,
            layout=layout,
            target=current_target,
            phase=phase,
            time_left=max(0.0, time_left),
            feedback_text=feedback_text,
            feedback_visible=feedback_visible,
            confidence=latest_confidence,
            hands_visible=hands_visible,
            xp=xp,
            xp_goal=XP_GOAL,
            target_color=target_color,
            last_attempt_label=last_attempt_label,
            candidate_label=candidate_label,
        )

        if phase == "REVIEW":
            review_frame = update_review_video(now)
            review_button = {}
            canvas = draw_review_overlay(canvas, review_target, review_frame, review_button)
            clicked = ui_state["click_pos"]
            if clicked is not None and point_in_rect(clicked[0], clicked[1], review_button):
                review_next_clicked = True
            ui_state["click_pos"] = None

        cv2.imshow(WINDOW_NAME, canvas)
        if window_closed(WINDOW_NAME):
            break

        # Tap/click anywhere on the footer card to start.
        if phase == "READY":
            clicked = ui_state["click_pos"]
            if clicked is not None:
                scale = layout["scale"]
                footer_x1, footer_y1, footer_x2, footer_y2 = layout["footer_rect"]
                card_pad = sp(10, scale)
                card_rect = {
                    "x1": footer_x1 + card_pad,
                    "y1": footer_y1 + card_pad,
                    "x2": footer_x2 - card_pad,
                    "y2": footer_y2 - card_pad,
                }
                if point_in_rect(clicked[0], clicked[1], card_rect):
                    phase = "ACTIVE"
                    round_start_time = time.time()
                    wrong_attempts_in_round = 0
                    reset_tracking_state()
                    if lesson_start_time is None:
                        lesson_start_time = now
            ui_state["click_pos"] = None

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("f"):
            window_state["fullscreen"] = not window_state["fullscreen"]
            apply_window_mode(WINDOW_NAME, window_state["fullscreen"])
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        if key in (32, ord("s")) and phase == "READY":
            phase = "ACTIVE"
            round_start_time = time.time()
            wrong_attempts_in_round = 0
            reset_tracking_state()
            if lesson_start_time is None:
                lesson_start_time = now

        if phase == "REVIEW" and (review_next_clicked or key in (32, 13)):
            review_next_clicked = False
            close_review_video()
            current_target = pending_target if pending_target else choose_target(ACTIONS, fail_counts)
            pending_target = None
            phase = "READY"
            wrong_attempts_in_round = 0
            round_start_time = 0.0
            reset_tracking_state()

        if lesson_completed:
            break

    if lesson_completed:
        failing = [(s, c) for s, c in fail_counts.items() if c > 0]
        failing.sort(key=lambda x: x[1], reverse=True)

        elapsed_seconds = None
        if lesson_start_time is not None and lesson_end_time is not None:
            elapsed_seconds = lesson_end_time - lesson_start_time

        print("\nLesson complete.")
        print(
            f"Correct: {success_count}, Wrong: {wrong_xp_count}, "
            f"Goal: {XP_GOAL}, Time: {format_duration(elapsed_seconds)}"
        )

        if failing:
            print("Needs practice:")
            for sign, count in failing:
                print(f"- {label_text(sign)}: {count}")
        else:
            print("No failed signs this lesson.")

        while True:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

            win_w, win_h = get_window_size(WINDOW_NAME, frame.shape[1], frame.shape[0])
            win_w = max(420, int(win_w))
            win_h = max(300, int(win_h))
            canvas = np.full((win_h, win_w, 3), rgb_to_bgr(PALETTE["mint"]), dtype=np.uint8)

            summary_button = {}
            canvas = draw_summary_screen(
                canvas,
                fail_counts=fail_counts,
                success_count=success_count,
                wrong_xp_count=wrong_xp_count,
                xp_goal=XP_GOAL,
                elapsed_seconds=elapsed_seconds,
                button=summary_button,
            )
            cv2.imshow(WINDOW_NAME, canvas)
            if window_closed(WINDOW_NAME):
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("f"):
                window_state["fullscreen"] = not window_state["fullscreen"]
                apply_window_mode(WINDOW_NAME, window_state["fullscreen"])
                cv2.setMouseCallback(WINDOW_NAME, on_mouse)
            clicked = ui_state["click_pos"]
            if clicked is not None and point_in_rect(clicked[0], clicked[1], summary_button):
                break
            ui_state["click_pos"] = None
finally:
    cap.release()
    close_review_video()
    if not KEEP_WINDOW_OPEN:
        cv2.destroyAllWindows()
    pose_landmarker.close()
    hand_landmarker.close()






















