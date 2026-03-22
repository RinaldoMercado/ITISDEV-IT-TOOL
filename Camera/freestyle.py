import os
import cv2
import time
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

# ManuMano Freestyle: live sentence building from sign predictions.


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
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\ambiguitytradition-regular.otf",
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\Ambiguity-Regular.otf",
    r"C:\Windows\Fonts\ambiguitytradition-regular.otf",
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
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\ambiguitytradition-bold.otf",
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\Ambiguity-Bold.otf",
    r"C:\Windows\Fonts\ambiguitytradition-bold.otf",
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


def open_camera(index=0, width=1280, height=720):
    """Open the webcam with low-latency settings."""
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


def label_text(label):
    return label.replace("_", " ").strip().title()


def human_sentence(words):
    if not words:
        return ""
    return " ".join(label_text(w) for w in words)


def truncate_text(text, max_chars=52):
    if len(text) <= max_chars:
        return text
    if max_chars < 6:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def wrap_words(text, max_chars, max_lines=3):
    if not text:
        return []

    words = text.split()
    lines = []
    current = ""

    for word in words:
        tentative = word if not current else f"{current} {word}"
        if len(tentative) <= max_chars:
            current = tentative
        else:
            if current:
                lines.append(current)
                current = word
            else:
                lines.append(word[:max_chars])
                current = word[max_chars:]

        if len(lines) >= max_lines:
            break

    if current and len(lines) < max_lines:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    if lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = truncate_text(lines[-1], max_chars=max_chars)

    return lines


# ---------------------------- PARAMETERS ----------------------------
POSE_MODEL = r"models/pose_landmarker.task"
HAND_MODEL = r"models/hand_landmarker.task"
LSTM_MODEL = r"action.h5"
DATA_PATH = "MP_Data"

SEQUENCE_LENGTH = 30
CONSISTENCY_FRAMES = 6
PREDICT_EVERY = 2
HAND_TIMEOUT = 1.2
DISPLAY_LOCK_TIME = 0.75
SENTENCE_BREAK_TIMEOUT = 1.0
REPEAT_WORD_DELAY = 1.6
WORD_HOLD_SECONDS = 1.2
HOLD_RESET_GRACE = 0.25


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

last_hand_seen_time = time.time()
frame_counter = 0

latest_candidate = ""
latest_confidence = 0.0
latest_model_prob = 0.0
latest_template_distance = 999.0

display_word = ""
display_time = 0.0

sentence_active = False
current_sentence_words = []
last_sentence_text = ""
sentence_history = []
last_appended_word = ""
last_append_time = 0.0
just_added_until = 0.0

hold_word = ""
hold_start_time = None
hold_last_seen_time = 0.0
hold_progress = 0.0

window_state = {"fullscreen": False}
WINDOW_NAME = os.environ.get("MANUMANO_WINDOW_NAME", get_css_str(STYLE_VARS, "window-freestyle", "ManuMano Freestyle"))
KEEP_WINDOW_OPEN = os.environ.get("MANUMANO_KEEP_WINDOW") == "1"


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
            "text": "ManuMano Freestyle",
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


# ---------------------------- CALLBACKS ----------------------------
def pose_callback(result, output_image, timestamp_ms):
    global latest_pose_result
    latest_pose_result = result.pose_landmarks


def hand_callback(result, output_image, timestamp_ms):
    global latest_hand_result, latest_handedness_result
    latest_hand_result = result.hand_landmarks
    latest_handedness_result = result.handedness


def reset_tracking_state():
    global latest_candidate, latest_confidence, latest_model_prob, latest_template_distance

    sequence.clear()
    predictions.clear()
    latest_candidate = ""
    latest_confidence = 0.0
    latest_model_prob = 0.0
    latest_template_distance = 999.0


def reset_hold_state():
    global hold_word, hold_start_time, hold_last_seen_time, hold_progress

    hold_word = ""
    hold_start_time = None
    hold_last_seen_time = 0.0
    hold_progress = 0.0


def finalize_sentence():
    global last_sentence_text, last_appended_word, last_append_time

    if current_sentence_words:
        sentence_text = human_sentence(current_sentence_words)
        last_sentence_text = sentence_text
        sentence_history.append(sentence_text)
        if len(sentence_history) > 4:
            sentence_history.pop(0)

    current_sentence_words.clear()
    last_appended_word = ""
    last_append_time = 0.0


def maybe_append_word(word, now):
    global last_appended_word, last_append_time, just_added_until

    if not word:
        return False

    should_append = False
    if not current_sentence_words:
        should_append = True
    elif word != last_appended_word:
        should_append = True
    elif now - last_append_time >= REPEAT_WORD_DELAY:
        should_append = True

    if not should_append:
        return False

    current_sentence_words.append(word)
    last_appended_word = word
    last_append_time = now
    just_added_until = now + 0.55
    return True


# ---------------------------- HUD ----------------------------
def draw_freestyle_hud(
    frame,
    display_word,
    confidence,
    hands_visible,
    current_words,
    last_sentence,
    sentence_count,
    just_added,
    hold_word,
    hold_progress,
):
    """Draw the freestyle HUD with a compact sentence card."""
    h, w, _ = frame.shape
    scale = get_ui_scale(w, h)

    header_h = max(sp(64, scale), int(h * 0.11))
    bottom_h = max(sp(94, scale), int(h * 0.16))
    bottom_y = h - bottom_h

    sentence_h = max(sp(60, scale), int(h * 0.1))
    sentence_x1 = sp(18, scale)
    sentence_x2 = w - sp(18, scale)
    sentence_y2 = bottom_y - sp(10, scale)
    sentence_y1 = sentence_y2 - sentence_h

    cv2.rectangle(frame, (0, 0), (w, header_h), BRAND_GREEN_BGR, -1)
    draw_panel(frame, sentence_x1, sentence_y1, sentence_x2, sentence_y2, CARD_WHITE_BGR, 0.96)
    draw_panel(frame, 0, bottom_y, w, h, CARD_WHITE_BGR, 0.95)
    cv2.rectangle(frame, (sentence_x1, sentence_y1), (sentence_x2, sentence_y2), CARD_GRAY_BGR, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, bottom_y), (w, h), CARD_GRAY_BGR, 2, cv2.LINE_AA)

    hold_bar_x1 = w - sp(190, scale)
    hold_bar_y1 = sp(40, scale)
    hold_bar_x2 = w - sp(18, scale)
    hold_bar_y2 = sp(54, scale)
    cv2.rectangle(frame, (hold_bar_x1, hold_bar_y1), (hold_bar_x2, hold_bar_y2), CARD_WHITE_BGR, -1)
    cv2.rectangle(frame, (hold_bar_x1, hold_bar_y1), (hold_bar_x2, hold_bar_y2), CARD_GRAY_BGR, 1, cv2.LINE_AA)
    cv2.rectangle(
        frame,
        (hold_bar_x1, hold_bar_y1),
        (hold_bar_x1 + int((hold_bar_x2 - hold_bar_x1) * max(0.0, min(1.0, hold_progress))), hold_bar_y2),
        BRAND_GREEN_SOFT_BGR,
        -1,
    )

    locked_word = label_text(display_word) if display_word else "Show A Sign"
    status_text = "Sentence active" if hands_visible else "Hands not detected"
    status_color = PALETTE["brand_green"] if hands_visible else PALETTE["text_muted"]

    current_sentence = human_sentence(current_words)
    has_sentence = bool(current_sentence)
    last_sentence_label = last_sentence if last_sentence else "-"

    hold_label = label_text(hold_word) if hold_word else "-"
    hold_hint = f"Hold: {hold_label}"
    match_percent = int(max(0.0, min(1.0, confidence)) * 100)

    max_chars_sentence = max(22, int(w / 16))
    sentence_lines = wrap_words(current_sentence, max_chars=max_chars_sentence, max_lines=2)

    texts = [
        {
            "text": "Freestyle",
            "pos": (w // 2, sp(14, scale)),
            "size": sp(22, scale, 14),
            "color": PALETTE["card_white"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": locked_word,
            "pos": (w // 2, header_h + sp(20, scale)),
            "size": sp(30, scale, 18),
            "color": PALETTE["brand_green"] if just_added or display_word else PALETTE["text_dark"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": hold_hint,
            "pos": (w - sp(18, scale), sp(18, scale)),
            "size": sp(14, scale, 11),
            "color": PALETTE["card_white"],
            "align": "right",
            "weight": "bold",
        },
        {
            "text": "Sentence",
            "pos": (sentence_x1 + sp(12, scale), sentence_y1 + sp(8, scale)),
            "size": sp(16, scale, 11),
            "color": PALETTE["text_dark"],
            "weight": "bold",
        },
        {
            "text": f"Completed: {sentence_count}",
            "pos": (sentence_x2 - sp(12, scale), sentence_y1 + sp(8, scale)),
            "size": sp(14, scale, 10),
            "color": PALETTE["text_muted"],
            "align": "right",
        },
        {
            "text": f"Last: {truncate_text(last_sentence_label, max_chars=max(24, int(w / 22)))}",
            "pos": (sentence_x1 + sp(12, scale), sentence_y2 - sp(18, scale)),
            "size": sp(14, scale, 10),
            "color": PALETTE["text_muted"],
        },
        {
            "text": f"Match: {match_percent}%",
            "pos": (sp(18, scale), bottom_y + sp(18, scale)),
            "size": sp(18, scale, 12),
            "color": PALETTE["text_dark"],
            "weight": "bold",
        },
        {
            "text": status_text,
            "pos": (w - sp(18, scale), bottom_y + sp(18, scale)),
            "size": sp(16, scale, 11),
            "color": status_color,
            "align": "right",
        },
        {
            "text": "ESC: close (or X)   F: fullscreen   C: clear",
            "pos": (w // 2, h - sp(24, scale)),
            "size": sp(13, scale, 10),
            "color": PALETTE["text_muted"],
            "align": "center",
        },
    ]

    if has_sentence and sentence_lines:
        line_y = sentence_y1 + sp(28, scale)
        step_y = sp(24, scale)
        for line in sentence_lines:
            texts.append(
                {
                    "text": line,
                    "pos": (w // 2, line_y),
                    "size": sp(22, scale, 14),
                    "color": PALETTE["brand_green"] if just_added else PALETTE["text_dark"],
                    "align": "center",
                    "weight": "bold",
                }
            )
            line_y += step_y
    else:
        texts.append(
            {
                "text": "Start signing to build a sentence",
                "pos": (w // 2, sentence_y1 + sp(28, scale)),
                "size": sp(16, scale, 12),
                "color": PALETTE["text_muted"],
                "align": "center",
            }
        )

    return draw_texts(frame, texts)


# ---------------------------- INIT ----------------------------
apply_window_mode(WINDOW_NAME, window_state["fullscreen"])

TOTAL_LOAD_STEPS = max(2, get_css_int(STYLE_VARS, "mode-loading-steps", 4))
draw_loading_screen(WINDOW_NAME, 1, TOTAL_LOAD_STEPS, "Preparing freestyle", "Initializing pose tracker")
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

draw_loading_screen(WINDOW_NAME, 2, TOTAL_LOAD_STEPS, "Preparing freestyle", "Initializing hand tracker")
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

draw_loading_screen(WINDOW_NAME, 3, TOTAL_LOAD_STEPS, "Preparing freestyle", "Loading LSTM model")
model = load_model(LSTM_MODEL, compile=False)

draw_loading_screen(WINDOW_NAME, 4, TOTAL_LOAD_STEPS, "Preparing freestyle", "Loading sign templates")
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

print("\nFreestyle controls: ESC or window X closes, F fullscreen, C clears text.\n")
start_time = time.time()


# ---------------------------- LOOP ----------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int((now - start_time) * 1000)

        pose_landmarker.detect_async(mp_image, timestamp)
        hand_landmarker.detect_async(mp_image, timestamp)

        left_hand, right_hand = split_hands_by_handedness(latest_hand_result, latest_handedness_result)
        hands_visible = left_hand is not None or right_hand is not None

        if hands_visible:
            if not sentence_active:
                sentence_active = True
                reset_tracking_state()
                reset_hold_state()
            last_hand_seen_time = now
        else:
            if sentence_active and (now - last_hand_seen_time > SENTENCE_BREAK_TIMEOUT):
                finalize_sentence()
                sentence_active = False
                display_word = ""
                reset_tracking_state()
                reset_hold_state()
            elif now - last_hand_seen_time > HAND_TIMEOUT:
                display_word = ""
                reset_tracking_state()
                reset_hold_state()

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

                    predictions.append(latest_candidate)
                    predictions = predictions[-CONSISTENCY_FRAMES:]


                    candidate_for_hold = ""
                    if predictions:
                        candidate_for_hold = max(set(predictions), key=predictions.count)
                    elif latest_candidate:
                        candidate_for_hold = latest_candidate

                    if candidate_for_hold:
                        if hold_word != candidate_for_hold:
                            hold_word = candidate_for_hold
                            hold_start_time = now
                            hold_last_seen_time = now
                            hold_progress = 0.0
                        else:
                            hold_last_seen_time = now
                            if hold_start_time is None:
                                hold_start_time = now
                            hold_progress = max(0.0, min(1.0, (now - hold_start_time) / WORD_HOLD_SECONDS))

                        if hold_progress >= 1.0 and now - display_time > DISPLAY_LOCK_TIME:
                            display_word = hold_word
                            display_time = now
                            maybe_append_word(display_word, now)
                            reset_hold_state()
                    elif hold_word and (now - hold_last_seen_time > HOLD_RESET_GRACE):
                        reset_hold_state()

        stage_frame, canvas, ox, oy = compose_display_frame(
            frame,
            WINDOW_NAME,
            rgb_to_bgr(PALETTE["mint"]),
        )

        stage_frame = draw_freestyle_hud(
            stage_frame,
            display_word=display_word,
            confidence=latest_confidence,
            hands_visible=hands_visible,
            current_words=current_sentence_words,
            last_sentence=last_sentence_text,
            sentence_count=len(sentence_history),
            just_added=(now < just_added_until),
            hold_word=hold_word,
            hold_progress=hold_progress,
        )

        display_frame = blit_stage(canvas, stage_frame, ox, oy)
        cv2.imshow(WINDOW_NAME, display_frame)
        if window_closed(WINDOW_NAME):
            break

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("f"):
            window_state["fullscreen"] = not window_state["fullscreen"]
            apply_window_mode(WINDOW_NAME, window_state["fullscreen"])
        if key == ord("c"):
            current_sentence_words.clear()
            sentence_history.clear()
            last_sentence_text = ""
            display_word = ""
            reset_tracking_state()
            reset_hold_state()

finally:
    cap.release()
    if not KEEP_WINDOW_OPEN:
        cv2.destroyAllWindows()
    pose_landmarker.close()
    hand_landmarker.close()




















