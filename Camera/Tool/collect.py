import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from gesture_matcher import split_hands_by_handedness, extract_keypoints

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ManuMano Collect: data collection tool with on-screen guidance.
# ----------------------------
# Visual Theme
# ----------------------------
PALETTE = {
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


def rgb_to_bgr(color):
    return (color[2], color[1], color[0])


BRAND_GREEN_BGR = rgb_to_bgr(PALETTE["brand_green"])
BRAND_GREEN_SOFT_BGR = rgb_to_bgr(PALETTE["brand_green_soft"])
BRAND_GREEN_DARK_BGR = rgb_to_bgr(PALETTE["brand_green_dark"])
CARD_WHITE_BGR = rgb_to_bgr(PALETTE["card_white"])
CARD_GRAY_BGR = rgb_to_bgr(PALETTE["card_gray"])
BLACK_BGR = rgb_to_bgr(PALETTE["black"])

REGULAR_FONT_CANDIDATES = [
    os.path.join("fonts", "inter", "Inter-Regular.otf"),
    os.path.join("fonts", "inter", "Inter-Medium.otf"),
    os.path.join("fonts", "ambiguitytradition-regular.otf"),
    os.path.join("fonts", "Ambiguity-Regular.otf"),
    os.path.join("fonts", "Ambiguity-Regular.ttf"),
    os.path.join("assets", "fonts", "ambiguitytradition-regular.otf"),
    os.path.join("assets", "fonts", "Ambiguity-Regular.otf"),
    os.path.join("assets", "fonts", "Ambiguity-Regular.ttf"),
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\ambiguitytradition-regular.otf",
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\Ambiguity-Regular.otf",
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\Ambiguity-Regular.ttf",
    "ambiguitytradition-regular.otf",
    "Ambiguity-Regular.otf",
    "Ambiguity-Regular.ttf",
]

BOLD_FONT_CANDIDATES = [
    os.path.join("fonts", "inter", "Inter-Bold.otf"),
    os.path.join("fonts", "inter", "Inter-SemiBold.otf"),
    os.path.join("fonts", "ambiguitytradition-bold.otf"),
    os.path.join("fonts", "Ambiguity-Bold.otf"),
    os.path.join("fonts", "Ambiguity-Bold.ttf"),
    os.path.join("assets", "fonts", "ambiguitytradition-bold.otf"),
    os.path.join("assets", "fonts", "Ambiguity-Bold.otf"),
    os.path.join("assets", "fonts", "Ambiguity-Bold.ttf"),
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\ambiguitytradition-bold.otf",
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\Ambiguity-Bold.otf",
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\Ambiguity-Bold.ttf",
    "ambiguitytradition-bold.otf",
    "Ambiguity-Bold.otf",
    "Ambiguity-Bold.ttf",
]

REGULAR_FONT_PATH = next((p for p in REGULAR_FONT_CANDIDATES if os.path.exists(p)), None)
BOLD_FONT_PATH = next((p for p in BOLD_FONT_CANDIDATES if os.path.exists(p)), None)
if not BOLD_FONT_PATH:
    BOLD_FONT_PATH = REGULAR_FONT_PATH

FONT_CACHE = {}


def get_font(size, weight="regular"):
    if not PIL_AVAILABLE:
        return None

    font_path = BOLD_FONT_PATH if weight == "bold" else REGULAR_FONT_PATH
    if not font_path:
        return None

    size = max(12, int(size))
    key = (weight, size)

    if key not in FONT_CACHE:
        try:
            FONT_CACHE[key] = ImageFont.truetype(font_path, size=size)
        except OSError:
            FONT_CACHE[key] = ImageFont.load_default()

    return FONT_CACHE[key]


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


def action_label(action):
    return action.replace("_", " ").title()


def parse_actions(text):
    return [
        word.strip().lower().replace(" ", "_")
        for word in text.split(",")
        if word.strip()
    ]


# ----------------------------
# Setup UI for sign input
# ----------------------------
SETUP_WINDOW = "ManuMano Setup"
COLLECTION_WINDOW = "ManuMano Collect"


def get_actions_from_ui():
    typed = ""
    setup_state = {"fullscreen": False}
    apply_window_mode(SETUP_WINDOW, setup_state["fullscreen"])

    fallback_w = 1280
    fallback_h = 720

    while True:
        setup_w, setup_h = get_window_size(SETUP_WINDOW, fallback_w, fallback_h)
        setup_w = max(720, int(setup_w))
        setup_h = max(420, int(setup_h))

        canvas = np.full((setup_h, setup_w, 3), rgb_to_bgr(PALETTE["mint"]), dtype=np.uint8)
        scale = get_ui_scale(setup_w, setup_h)

        card_w = min(setup_w - sp(40, scale, 28), int(setup_w * 0.88))
        card_h = min(setup_h - sp(36, scale, 24), int(setup_h * 0.82))
        card_w = max(card_w, 520)
        card_h = max(card_h, 320)

        card_x1 = (setup_w - card_w) // 2
        card_y1 = (setup_h - card_h) // 2
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h

        draw_panel(canvas, card_x1, card_y1, card_x2, card_y2, CARD_WHITE_BGR, 0.95)
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), CARD_GRAY_BGR, 2, cv2.LINE_AA)

        input_x1 = card_x1 + sp(40, scale)
        input_x2 = card_x2 - sp(40, scale)
        input_y1 = card_y1 + sp(148, scale)
        input_y2 = input_y1 + sp(72, scale)

        draw_panel(canvas, input_x1, input_y1, input_x2, input_y2, CARD_GRAY_BGR, 0.75)
        cv2.rectangle(canvas, (input_x1, input_y1), (input_x2, input_y2), BRAND_GREEN_BGR, 2, cv2.LINE_AA)

        max_chars = max(48, int(setup_w * 0.08))
        preview = typed if typed else "Type signs here..."
        preview = preview[-max_chars:]

        canvas = draw_texts(
            canvas,
            [
                {
                    "text": "ManuMano",
                    "pos": (setup_w // 2, card_y1 + sp(24, scale)),
                    "size": sp(50, scale, 20),
                    "color": PALETTE["brand_green"],
                    "align": "center",
                    "weight": "bold",
                },
                {
                    "text": "What signs should we collect?",
                    "pos": (setup_w // 2, card_y1 + sp(82, scale)),
                    "size": sp(32, scale, 16),
                    "color": PALETTE["text_dark"],
                    "align": "center",
                    "weight": "bold",
                },
                {
                    "text": "Use commas between signs. Example: i love you, thank you",
                    "pos": (setup_w // 2, card_y1 + sp(118, scale)),
                    "size": sp(20, scale, 11),
                    "color": PALETTE["text_muted"],
                    "align": "center",
                },
                {
                    "text": preview,
                    "pos": (input_x1 + sp(16, scale), input_y1 + sp(18, scale)),
                    "size": sp(30, scale, 14),
                    "color": PALETTE["brand_green" if typed else "text_muted"],
                    "weight": "bold" if typed else "regular",
                },
                {
                    "text": "ENTER: Continue   |   BACKSPACE: Delete   |   F: Fullscreen   |   ESC: Cancel",
                    "pos": (setup_w // 2, card_y2 - sp(52, scale)),
                    "size": sp(19, scale, 11),
                    "color": PALETTE["text_muted"],
                    "align": "center",
                },
            ],
        )

        cv2.imshow(SETUP_WINDOW, canvas)
        if window_closed(SETUP_WINDOW):
            return []

        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            actions = parse_actions(typed)
            if actions:
                cv2.destroyWindow(SETUP_WINDOW)
                return actions

        elif key == 27:
            cv2.destroyWindow(SETUP_WINDOW)
            return []

        elif key in (8, 127):
            typed = typed[:-1]

        elif key == ord("f"):
            setup_state["fullscreen"] = not setup_state["fullscreen"]
            apply_window_mode(SETUP_WINDOW, setup_state["fullscreen"])

        elif 32 <= key <= 126:
            ch = chr(key)
            if ch.isalnum() or ch in [" ", ",", "_", "-", "'"]:
                typed += ch


# ----------------------------
# Collect parameters
# ----------------------------
POSE_MODEL = r"models/pose_landmarker.task"
HAND_MODEL = r"models/hand_landmarker.task"
DATA_PATH = "MP_Data"
VIDEO_BASE_PATH = "FSL Videos"
VIDEO_EXT = ".mp4"
VIDEO_CODEC = "mp4v"

SEQUENCE_LENGTH = 30
SEQUENCES_PER_ACTION = 5
EXTRA_FRAMES_PER_EXTRA_WORD = 14
MAX_CAPTURE_WINDOW = 64

FRAME_DELAY = 20
VISIBILITY_THRESHOLD = 0.5
MAX_RETAKES_PER_SEQUENCE = 3

MIN_HAND_RATIO_SINGLE = 0.52
MIN_HAND_RATIO_PHRASE = 0.62
MIN_BOTH_RATIO_SINGLE = 0.05
MIN_BOTH_RATIO_PHRASE = 0.20

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


def get_capture_window(action):
    words = max(1, len(action.split("_")))
    window = SEQUENCE_LENGTH + (words - 1) * EXTRA_FRAMES_PER_EXTRA_WORD
    return min(MAX_CAPTURE_WINDOW, max(SEQUENCE_LENGTH, window))


def quality_targets(action, capture_window):
    words = max(1, len(action.split("_")))
    if words > 1:
        min_hands = int(capture_window * MIN_HAND_RATIO_PHRASE)
        min_both = int(capture_window * MIN_BOTH_RATIO_PHRASE)
    else:
        min_hands = int(capture_window * MIN_HAND_RATIO_SINGLE)
        min_both = int(capture_window * MIN_BOTH_RATIO_SINGLE)
    return min_hands, min_both


def temporal_resample(frames, target_len):
    if len(frames) == target_len:
        return np.array(frames)

    idx = np.linspace(0, len(frames) - 1, target_len).round().astype(int)
    return np.array(frames)[idx]


def draw_hand(frame, hand, point_color, line_color):
    if hand is None:
        return

    h, w, _ = frame.shape
    radius = max(2, int(min(h, w) * 0.005))

    for lm in hand:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), radius, point_color, -1)

    for a, b in HAND_CONNECTIONS:
        p1, p2 = hand[a], hand[b]
        cv2.line(
            frame,
            (int(p1.x * w), int(p1.y * h)),
            (int(p2.x * w), int(p2.y * h)),
            line_color,
            2,
            cv2.LINE_AA,
        )


def draw_landmarks(frame, pose_landmarks, left_hand, right_hand):
    h, w, _ = frame.shape
    radius = max(2, int(min(h, w) * 0.005))

    if pose_landmarks:
        for pose in pose_landmarks:
            for lm in pose:
                if getattr(lm, "visibility", 1) > VISIBILITY_THRESHOLD:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), radius, BRAND_GREEN_SOFT_BGR, -1)

            for a, b in POSE_CONNECTIONS:
                p1, p2 = pose[a], pose[b]
                if (
                    getattr(p1, "visibility", 1) > VISIBILITY_THRESHOLD
                    and getattr(p2, "visibility", 1) > VISIBILITY_THRESHOLD
                ):
                    cv2.line(
                        frame,
                        (int(p1.x * w), int(p1.y * h)),
                        (int(p2.x * w), int(p2.y * h)),
                        BRAND_GREEN_DARK_BGR,
                        2,
                        cv2.LINE_AA,
                    )

    draw_hand(frame, left_hand, BRAND_GREEN_SOFT_BGR, BRAND_GREEN_BGR)
    draw_hand(frame, right_hand, CARD_GRAY_BGR, BRAND_GREEN_DARK_BGR)


def draw_collection_hud(
    frame,
    action,
    seq_num,
    attempt,
    frame_count,
    capture_window,
    hands_visible,
    both_visible,
    hands_frames,
    both_frames,
    min_hands_frames,
    min_both_frames,
):
    h, w, _ = frame.shape
    scale = get_ui_scale(w, h)

    margin = sp(12, scale, 8)
    top_h = max(sp(86, scale), int(h * 0.14))
    bottom_h = max(sp(112, scale), int(h * 0.18))
    bottom_y = h - bottom_h

    draw_panel(frame, margin, margin, w - margin, top_h, CARD_WHITE_BGR, 0.9)
    draw_panel(frame, margin, bottom_y, w - margin, h - margin, CARD_WHITE_BGR, 0.9)
    cv2.rectangle(frame, (margin, margin), (w - margin, top_h), CARD_GRAY_BGR, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (margin, bottom_y), (w - margin, h - margin), CARD_GRAY_BGR, 2, cv2.LINE_AA)

    bar_margin = sp(30, scale)
    bar_y1 = h - sp(36, scale)
    bar_y2 = h - sp(18, scale)
    cv2.rectangle(frame, (bar_margin, bar_y1), (w - bar_margin, bar_y2), CARD_GRAY_BGR, -1)

    progress = frame_count / float(capture_window)
    fill_w = int((w - 2 * bar_margin) * progress)
    cv2.rectangle(frame, (bar_margin, bar_y1), (bar_margin + fill_w, bar_y2), BRAND_GREEN_BGR, -1)

    if both_visible:
        status_text = "Both hands visible"
        status_color = PALETTE["brand_green"]
    elif hands_visible:
        status_text = "One hand visible"
        status_color = PALETTE["brand_green_dark"]
    else:
        status_text = "Show your hands"
        status_color = PALETTE["text_muted"]

    texts = [
        {
            "text": "ManuMano",
            "pos": (sp(24, scale), sp(18, scale)),
            "size": sp(28, scale, 16),
            "color": PALETTE["brand_green"],
            "weight": "bold",
        },
        {
            "text": f"Collect: {action_label(action)}",
            "pos": (sp(24, scale), sp(46, scale)),
            "size": sp(22, scale, 14),
            "color": PALETTE["text_dark"],
            "weight": "bold",
        },
        {
            "text": f"Seq {seq_num}   Attempt {attempt}/{MAX_RETAKES_PER_SEQUENCE}",
            "pos": (w - sp(24, scale), sp(22, scale)),
            "size": sp(20, scale, 12),
            "color": PALETTE["text_muted"],
            "align": "right",
        },
        {
            "text": f"Raw {frame_count}/{capture_window}",
            "pos": (w - sp(24, scale), sp(50, scale)),
            "size": sp(20, scale, 12),
            "color": PALETTE["text_muted"],
            "align": "right",
        },
        {
            "text": status_text,
            "pos": (sp(24, scale), bottom_y + sp(14, scale)),
            "size": sp(22, scale, 13),
            "color": status_color,
            "weight": "bold",
        },
        {
            "text": f"Hands: {hands_frames}/{min_hands_frames}   Both: {both_frames}/{min_both_frames}",
            "pos": (sp(24, scale), bottom_y + sp(42, scale)),
            "size": sp(20, scale, 12),
            "color": PALETTE["text_dark"],
        },
        {
            "text": "ESC: close (or X)   F: fullscreen",
            "pos": (w - sp(24, scale), bottom_y + sp(42, scale)),
            "size": sp(18, scale, 12),
            "color": PALETTE["text_muted"],
            "align": "right",
        },
    ]

    return draw_texts(frame, texts)


def draw_countdown(frame, action, seq_num, attempt, countdown_value):
    h, w, _ = frame.shape
    scale = get_ui_scale(w, h)

    draw_panel(frame, 0, 0, w, h, rgb_to_bgr(PALETTE["mint"]), 0.25)

    card_w = int(w * 0.68)
    card_h = int(h * 0.52)
    x1 = (w - card_w) // 2
    y1 = (h - card_h) // 2
    x2 = x1 + card_w
    y2 = y1 + card_h

    draw_panel(frame, x1, y1, x2, y2, CARD_WHITE_BGR, 0.95)
    cv2.rectangle(frame, (x1, y1), (x2, y2), CARD_GRAY_BGR, 2, cv2.LINE_AA)

    texts = [
        {
            "text": "ManuMano",
            "pos": (w // 2, y1 + sp(20, scale)),
            "size": sp(34, scale, 16),
            "color": PALETTE["brand_green"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": "Get ready to sign",
            "pos": (w // 2, y1 + sp(70, scale)),
            "size": sp(24, scale, 12),
            "color": PALETTE["text_dark"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": action_label(action),
            "pos": (w // 2, y1 + sp(110, scale)),
            "size": sp(30, scale, 14),
            "color": PALETTE["brand_green_dark"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": str(countdown_value),
            "pos": (w // 2, y1 + sp(160, scale)),
            "size": sp(88, scale, 28),
            "color": PALETTE["brand_green"],
            "align": "center",
            "weight": "bold",
        },
        {
            "text": f"Seq {seq_num}   Attempt {attempt}/{MAX_RETAKES_PER_SEQUENCE}",
            "pos": (w // 2, y2 - sp(42, scale)),
            "size": sp(20, scale, 12),
            "color": PALETTE["text_muted"],
            "align": "center",
        },
    ]

    return draw_texts(frame, texts)


def countdown(cap, action, seq_num, attempt, window_state):
    for count in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            return False

        stage_frame, canvas, ox, oy = compose_display_frame(
            frame,
            COLLECTION_WINDOW,
            rgb_to_bgr(PALETTE["mint"]),
        )
        stage_frame = draw_countdown(stage_frame, action, seq_num, attempt, count)
        display_frame = blit_stage(canvas, stage_frame, ox, oy)

        cv2.imshow(COLLECTION_WINDOW, display_frame)
        if window_closed(COLLECTION_WINDOW):
            return False

        key = cv2.waitKey(1000) & 0xFF
        if key == 27:
            return False
        if key == ord("f"):
            window_state["fullscreen"] = not window_state["fullscreen"]
            apply_window_mode(COLLECTION_WINDOW, window_state["fullscreen"])

    return True


def draw_loading_screen(progress, status_text, window_state):
    setup_w, setup_h = get_window_size(COLLECTION_WINDOW, 1280, 720)
    setup_w = max(720, int(setup_w))
    setup_h = max(420, int(setup_h))

    canvas = np.full((setup_h, setup_w, 3), rgb_to_bgr(PALETTE["mint"]), dtype=np.uint8)
    scale = get_ui_scale(setup_w, setup_h)

    card_w = min(setup_w - sp(40, scale, 28), int(setup_w * 0.82))
    card_h = min(setup_h - sp(36, scale, 24), int(setup_h * 0.58))
    card_w = max(card_w, 520)
    card_h = max(card_h, 280)

    card_x1 = (setup_w - card_w) // 2
    card_y1 = (setup_h - card_h) // 2
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h

    draw_panel(canvas, card_x1, card_y1, card_x2, card_y2, CARD_WHITE_BGR, 0.95)
    cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), CARD_GRAY_BGR, 2, cv2.LINE_AA)

    bar_x1 = card_x1 + sp(52, scale)
    bar_x2 = card_x2 - sp(52, scale)
    bar_y1 = card_y1 + sp(162, scale)
    bar_y2 = bar_y1 + sp(26, scale)

    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), CARD_GRAY_BGR, -1)
    fill_w = int((bar_x2 - bar_x1) * max(0.0, min(1.0, progress)))
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), BRAND_GREEN_BGR, -1)

    canvas = draw_texts(
        canvas,
        [
            {
                "text": "ManuMano",
                "pos": (setup_w // 2, card_y1 + sp(28, scale)),
                "size": sp(50, scale, 20),
                "color": PALETTE["brand_green"],
                "align": "center",
                "weight": "bold",
            },
            {
                "text": "Preparing Collection Session",
                "pos": (setup_w // 2, card_y1 + sp(88, scale)),
                "size": sp(30, scale, 15),
                "color": PALETTE["text_dark"],
                "align": "center",
                "weight": "bold",
            },
            {
                "text": status_text,
                "pos": (setup_w // 2, card_y1 + sp(122, scale)),
                "size": sp(22, scale, 12),
                "color": PALETTE["brand_green_dark"],
                "align": "center",
            },
            {
                "text": f"{int(progress * 100)}%",
                "pos": (setup_w // 2, bar_y2 + sp(12, scale)),
                "size": sp(24, scale, 12),
                "color": PALETTE["text_dark"],
                "align": "center",
                "weight": "bold",
            },
            {
                "text": "F: fullscreen   |   ESC: close",
                "pos": (setup_w // 2, card_y2 - sp(42, scale)),
                "size": sp(18, scale, 11),
                "color": PALETTE["text_muted"],
                "align": "center",
            },
        ],
    )

    cv2.imshow(COLLECTION_WINDOW, canvas)


def show_loading_step(progress, status_text, window_state, wait_ms=80):
    draw_loading_screen(progress, status_text, window_state)

    end = time.time() + (wait_ms / 1000.0)
    while time.time() < end:
        if window_closed(COLLECTION_WINDOW):
            return False

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            return False
        if key == ord("f"):
            window_state["fullscreen"] = not window_state["fullscreen"]
            apply_window_mode(COLLECTION_WINDOW, window_state["fullscreen"])
            draw_loading_screen(progress, status_text, window_state)

    return True
# ----------------------------
# User input via UI
# ----------------------------
ACTIONS = get_actions_from_ui()

if len(ACTIONS) == 0:
    print("No signs entered. Exiting.")
    raise SystemExit

if REGULAR_FONT_PATH:
    print(f"Using ManuMano regular font: {REGULAR_FONT_PATH}")
if BOLD_FONT_PATH:
    print(f"Using ManuMano bold font: {BOLD_FONT_PATH}")


# ----------------------------
# Initialize resources with loading screen
# ----------------------------
cv2.setUseOptimized(True)
loading_state = {"fullscreen": False}
apply_window_mode(COLLECTION_WINDOW, loading_state["fullscreen"])

if not show_loading_step(0.12, "Preparing folders...", loading_state):
    raise SystemExit

os.makedirs(DATA_PATH, exist_ok=True)
for action in ACTIONS:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)
os.makedirs(VIDEO_BASE_PATH, exist_ok=True)
for action in ACTIONS:
    os.makedirs(os.path.join(VIDEO_BASE_PATH, action), exist_ok=True)

if not show_loading_step(0.42, "Loading pose detector...", loading_state):
    raise SystemExit

pose_landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
)

if not show_loading_step(0.66, "Loading hand detector...", loading_state):
    pose_landmarker.close()
    raise SystemExit

hand_landmarker = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
)

if not show_loading_step(0.85, "Opening camera...", loading_state):
    pose_landmarker.close()
    hand_landmarker.close()
    raise SystemExit

cap = open_camera()
if cap is None:
    pose_landmarker.close()
    hand_landmarker.close()
    raise SystemExit

if not show_loading_step(1.0, "Ready!", loading_state, wait_ms=120):
    cap.release()
    pose_landmarker.close()
    hand_landmarker.close()
    raise SystemExit

window_state = loading_state
apply_window_mode(COLLECTION_WINDOW, window_state["fullscreen"])
print("\nPress ESC or close window anytime to stop. Press F to toggle fullscreen.\n")

perf_start = time.perf_counter()

# ----------------------------
# Loop
# ----------------------------
try:
    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)

        existing_files = []
        for name in os.listdir(action_path):
            if name.endswith(".npy"):
                try:
                    existing_files.append(int(name.split("_")[0]))
                except ValueError:
                    continue

        start_seq = max(existing_files) + 1 if existing_files else 1

        print(f"\nCollecting for '{action}'")
        print(f"Starting from sequence {start_seq}")

        for seq_num in range(start_seq, start_seq + SEQUENCES_PER_ACTION):
            capture_window = get_capture_window(action)
            min_hands_frames, min_both_frames = quality_targets(action, capture_window)

            saved = False
            attempt = 1

            while not saved and attempt <= MAX_RETAKES_PER_SEQUENCE:
                if not countdown(cap, action, seq_num, attempt, window_state):
                    raise KeyboardInterrupt

                # Record the raw camera stream for lesson-mode video hints.
                video_dir = os.path.join(VIDEO_BASE_PATH, action)
                temp_video_path = os.path.join(
                    video_dir,
                    f"seq_{seq_num}_attempt_{attempt}.tmp{VIDEO_EXT}",
                )
                final_video_path = os.path.join(video_dir, f"seq_{seq_num}{VIDEO_EXT}")
                video_writer = None
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                if video_fps <= 1:
                    video_fps = 30.0

                raw_sequence = []
                hands_frames = 0
                both_frames = 0

                for raw_idx in range(capture_window):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if video_writer is None:
                        h, w = frame.shape[:2]
                        video_writer = cv2.VideoWriter(
                            temp_video_path,
                            cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                            video_fps,
                            (w, h),
                        )
                        if not video_writer.isOpened():
                            video_writer.release()
                            video_writer = None

                    if video_writer is not None:
                        video_writer.write(frame)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp_ms = int((time.perf_counter() - perf_start) * 1000)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

                    pose_landmarks = pose_result.pose_landmarks if pose_result else None
                    hand_landmarks = hand_result.hand_landmarks if hand_result else None
                    handedness = hand_result.handedness if hand_result else None

                    left_hand, right_hand = split_hands_by_handedness(hand_landmarks, handedness)

                    hands_visible = left_hand is not None or right_hand is not None
                    both_visible = left_hand is not None and right_hand is not None

                    if hands_visible:
                        hands_frames += 1
                    if both_visible:
                        both_frames += 1

                    keypoints = extract_keypoints(pose_landmarks, left_hand, right_hand)
                    raw_sequence.append(keypoints)

                    stage_frame, canvas, ox, oy = compose_display_frame(
                        frame,
                        COLLECTION_WINDOW,
                        rgb_to_bgr(PALETTE["mint"]),
                    )

                    draw_landmarks(stage_frame, pose_landmarks, left_hand, right_hand)
                    stage_frame = draw_collection_hud(
                        stage_frame,
                        action=action,
                        seq_num=seq_num,
                        attempt=attempt,
                        frame_count=raw_idx + 1,
                        capture_window=capture_window,
                        hands_visible=hands_visible,
                        both_visible=both_visible,
                        hands_frames=hands_frames,
                        both_frames=both_frames,
                        min_hands_frames=min_hands_frames,
                        min_both_frames=min_both_frames,
                    )

                    display_frame = blit_stage(canvas, stage_frame, ox, oy)
                    cv2.imshow(COLLECTION_WINDOW, display_frame)
                    if window_closed(COLLECTION_WINDOW):
                        raise KeyboardInterrupt

                    key = cv2.waitKey(FRAME_DELAY) & 0xFF
                    if key == 27:
                        raise KeyboardInterrupt
                    if key == ord("f"):
                        window_state["fullscreen"] = not window_state["fullscreen"]
                        apply_window_mode(COLLECTION_WINDOW, window_state["fullscreen"])

                if video_writer is not None:
                    video_writer.release()

                if len(raw_sequence) != capture_window:
                    print("Frame read interrupted, retaking sequence.")
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    attempt += 1
                    continue

                quality_ok = hands_frames >= min_hands_frames and both_frames >= min_both_frames
                sampled_sequence = temporal_resample(raw_sequence, SEQUENCE_LENGTH)

                if quality_ok or attempt == MAX_RETAKES_PER_SEQUENCE:
                    for frame_idx, keypoints in enumerate(sampled_sequence):
                        np.save(os.path.join(action_path, f"{seq_num}_{frame_idx}"), keypoints)

                    if os.path.exists(temp_video_path):
                        try:
                            os.replace(temp_video_path, final_video_path)
                        except OSError:
                            os.remove(temp_video_path)

                    if quality_ok:
                        print(
                            f"Saved sequence {seq_num} (quality ok, raw {capture_window} -> {SEQUENCE_LENGTH})"
                        )
                    else:
                        print(
                            f"Saved sequence {seq_num} on last attempt (raw {capture_window} -> {SEQUENCE_LENGTH})"
                        )
                        print(
                            f"  Visibility below target: hands {hands_frames}/{min_hands_frames}, both {both_frames}/{min_both_frames}"
                        )

                    saved = True
                else:
                    print(
                        f"Retake {attempt + 1}/{MAX_RETAKES_PER_SEQUENCE} for seq {seq_num} "
                        f"(hands {hands_frames}/{min_hands_frames}, both {both_frames}/{min_both_frames})"
                    )
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    attempt += 1

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()
    hand_landmarker.close()

print("\nDone collecting.\n")


