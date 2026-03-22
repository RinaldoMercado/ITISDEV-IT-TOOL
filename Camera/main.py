import os
import cv2
import runpy
import numpy as np

from ui_style import (
    load_style_vars,
    palette_from_vars,
    resolve_font_candidates,
    get_css_int,
    get_css_float,
    get_css_str,
)

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ManuMano main menu: choose Freestyle or Lesson in a single window.


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


REGULAR_FONT_DEFAULTS = [
    os.path.join("fonts", "inter", "Inter-Regular.otf"),
    os.path.join("fonts", "inter", "Inter-Medium.otf"),
    os.path.join("fonts", "ambiguitytradition-regular.otf"),
    os.path.join("fonts", "Ambiguity-Regular.otf"),
    os.path.join("fonts", "Ambiguity-Regular.ttf"),
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\ambiguitytradition-regular.otf",
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\Ambiguity-Regular.otf",
]

BOLD_FONT_DEFAULTS = [
    os.path.join("fonts", "inter", "Inter-Bold.otf"),
    os.path.join("fonts", "inter", "Inter-SemiBold.otf"),
    os.path.join("fonts", "ambiguitytradition-bold.otf"),
    os.path.join("fonts", "Ambiguity-Bold.otf"),
    os.path.join("fonts", "Ambiguity-Bold.ttf"),
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\ambiguitytradition-bold.otf",
    r"C:\Users\Dane\OneDrive\Documents\ITISHCI\fonts\Ambiguity-Bold.otf",
]

REGULAR_FONT_CANDIDATES = resolve_font_candidates(STYLE_VARS, "font-regular", REGULAR_FONT_DEFAULTS)
BOLD_FONT_CANDIDATES = resolve_font_candidates(STYLE_VARS, "font-bold", BOLD_FONT_DEFAULTS)

REGULAR_FONT_PATH = next((p for p in REGULAR_FONT_CANDIDATES if os.path.exists(p)), None)
BOLD_FONT_PATH = next((p for p in BOLD_FONT_CANDIDATES if os.path.exists(p)), None)
if not BOLD_FONT_PATH:
    BOLD_FONT_PATH = REGULAR_FONT_PATH

FONT_CACHE = {}

MENU_OUTER_MARGIN_RATIO = get_css_float(STYLE_VARS, "menu-outer-margin-ratio", 0.05)
MENU_GAP_RATIO = get_css_float(STYLE_VARS, "menu-gap-ratio", 0.03)
MENU_CARD_MAX_HEIGHT = get_css_int(STYLE_VARS, "menu-card-max-height", 190)
MENU_CARD_Y_RATIO = get_css_float(STYLE_VARS, "menu-card-y-ratio", 0.38)
MENU_BUTTON_RADIUS = max(8, get_css_int(STYLE_VARS, "menu-button-radius", 34))
MENU_LOADING_STEPS = max(2, get_css_int(STYLE_VARS, "menu-loading-steps", 9))


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
        cv2.putText(frame, text, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, scale, rgb_to_bgr(PALETTE["card_white"]), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness, cv2.LINE_AA)

    return frame




def fill_round_rect(frame, x1, y1, x2, y2, radius, color):
    radius = max(6, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)


def get_window_size(window_name, fallback_w, fallback_h):
    try:
        _, _, w, h = cv2.getWindowImageRect(window_name)
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        pass
    return fallback_w, fallback_h


def build_buttons(w, h):
    outer_margin = max(40, int(w * MENU_OUTER_MARGIN_RATIO))
    gap = max(20, int(w * MENU_GAP_RATIO))
    usable_w = max(560, w - (2 * outer_margin) - gap)
    card_w = max(250, usable_w // 2)
    card_h = int(min(MENU_CARD_MAX_HEIGHT, h * 0.27))

    x0 = max(outer_margin, (w - ((2 * card_w) + gap)) // 2)
    y0 = int(h * MENU_CARD_Y_RATIO)

    return [
        {
            "id": "freestyle",
            "label": "Freestyle",
            "subtitle": "Build live sentences",
            "script": "freestyle.py",
            "x1": x0,
            "y1": y0,
            "x2": x0 + card_w,
            "y2": y0 + card_h,
        },
        {
            "id": "lesson",
            "label": "Lesson",
            "subtitle": "Guided sign challenge",
            "script": "lesson.py",
            "x1": x0 + card_w + gap,
            "y1": y0,
            "x2": x0 + card_w + gap + card_w,
            "y2": y0 + card_h,
        },
    ]


def point_in_button(x, y, button):
    return button["x1"] <= x <= button["x2"] and button["y1"] <= y <= button["y2"]


def run_mode_in_process(base_dir, script_name, window_name):
    script_path = os.path.join(base_dir, script_name)
    if not os.path.exists(script_path):
        print(f"Missing script: {script_path}")
        return

    prev_name = os.environ.get("MANUMANO_WINDOW_NAME")
    prev_keep = os.environ.get("MANUMANO_KEEP_WINDOW")
    prev_cwd = os.getcwd()

    os.environ["MANUMANO_WINDOW_NAME"] = window_name
    os.environ["MANUMANO_KEEP_WINDOW"] = "1"

    print(f"Launching {script_name}...")
    try:
        os.chdir(base_dir)
        runpy.run_path(script_path, run_name="__main__")
    except SystemExit:
        pass
    finally:
        if prev_name is None:
            os.environ.pop("MANUMANO_WINDOW_NAME", None)
        else:
            os.environ["MANUMANO_WINDOW_NAME"] = prev_name

        if prev_keep is None:
            os.environ.pop("MANUMANO_KEEP_WINDOW", None)
        else:
            os.environ["MANUMANO_KEEP_WINDOW"] = prev_keep

        try:
            os.chdir(prev_cwd)
        except OSError:
            pass


def show_mode_loading(window_name, mode_label):
    win_w, win_h = get_window_size(window_name, 1100, 680)
    win_w = max(780, win_w)
    win_h = max(480, win_h)

    for step in range(1, MENU_LOADING_STEPS + 1):
        frame = np.full((win_h, win_w, 3), rgb_to_bgr(PALETTE["mint"]), dtype=np.uint8)

        margin = 28
        draw_panel(frame, margin, margin, win_w - margin, win_h - margin, rgb_to_bgr(PALETTE["card_white"]), 0.92)
        cv2.rectangle(frame, (margin, margin), (win_w - margin, win_h - margin), rgb_to_bgr(PALETTE["card_gray"]), 2, cv2.LINE_AA)

        bar_x1 = 96
        bar_y1 = int(win_h * 0.64)
        bar_y2 = bar_y1 + 24
        ratio = step / float(MENU_LOADING_STEPS)
        cv2.rectangle(frame, (bar_x1, bar_y1), (win_w - bar_x1, bar_y2), rgb_to_bgr(PALETTE["card_gray"]), -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + int((win_w - (2 * bar_x1)) * ratio), bar_y2), rgb_to_bgr(PALETTE["brand_green"]), -1)

        frame = draw_texts(
            frame,
            [
                {
                    "text": "ManuMano",
                    "pos": (win_w // 2, int(win_h * 0.26)),
                    "size": 62,
                    "color": PALETTE["brand_green"],
                    "align": "center",
                    "weight": "bold",
                },
                {
                    "text": f"Opening {mode_label}",
                    "pos": (win_w // 2, int(win_h * 0.42)),
                    "size": 32,
                    "color": PALETTE["text_dark"],
                    "align": "center",
                    "weight": "bold",
                },
                {
                    "text": "Starting loading screen...",
                    "pos": (win_w // 2, int(win_h * 0.50)),
                    "size": 22,
                    "color": PALETTE["text_muted"],
                    "align": "center",
                },
            ],
        )

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

WINDOW_NAME = get_css_str(STYLE_VARS, "window-main", "ManuMano")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
window_state = {"fullscreen": False}
ui_state = {
    "mouse_pos": (-1, -1),
    "click_pos": None,
}


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


apply_window_mode(WINDOW_NAME, window_state["fullscreen"])
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

print("\nMain menu controls: click a button, or press 1/2. ESC or window X closes.\n")

try:
    while True:
        win_w, win_h = get_window_size(WINDOW_NAME, 1100, 680)
        win_w = max(780, win_w)
        win_h = max(480, win_h)

        frame = np.full((win_h, win_w, 3), rgb_to_bgr(PALETTE["mint"]), dtype=np.uint8)

        top_h = int(win_h * 0.18)
        cv2.rectangle(frame, (0, 0), (win_w, top_h), rgb_to_bgr(PALETTE["brand_green"]), -1)

        buttons = build_buttons(win_w, win_h)

        mx, my = ui_state["mouse_pos"]
        clicked = ui_state["click_pos"]

        for button in buttons:
            is_hover = point_in_button(mx, my, button)

            bg = rgb_to_bgr(PALETTE["brand_green_soft"] if is_hover else PALETTE["card_white"])
            fill_round_rect(frame, button["x1"], button["y1"], button["x2"], button["y2"], radius=MENU_BUTTON_RADIUS, color=bg)
            border = rgb_to_bgr(PALETTE["brand_green"]) if is_hover else rgb_to_bgr(PALETTE["card_gray"])
            cv2.rectangle(frame, (button["x1"], button["y1"]), (button["x2"], button["y2"]), border, 2, cv2.LINE_AA)
            cx = (button["x1"] + button["x2"]) // 2
            cy = (button["y1"] + button["y2"]) // 2
            frame = draw_texts(
                frame,
                [
                    {
                        "text": button["label"],
                        "pos": (cx, cy - 24),
                        "size": 40,
                        "color": PALETTE["brand_green_dark"],
                        "align": "center",
                        "weight": "bold",
                    },
                    {
                        "text": button["subtitle"],
                        "pos": (cx, cy + 14),
                        "size": 23,
                        "color": PALETTE["text_dark"],
                        "align": "center",
                    },
                ],
            )

        frame = draw_texts(
            frame,
            [
                {
                    "text": "ManuMano",
                    "pos": (win_w // 2, 28),
                    "size": 64,
                    "color": PALETTE["card_white"],
                    "align": "center",
                    "weight": "bold",
                },
                {
                    "text": "Choose a mode",
                    "pos": (win_w // 2, 92),
                    "size": 26,
                    "color": PALETTE["card_white"],
                    "align": "center",
                },
                {
                    "text": "1: Freestyle   2: Lesson   F: fullscreen   ESC: close",
                    "pos": (win_w // 2, win_h - 48),
                    "size": 22,
                    "color": PALETTE["text_dark"],
                    "align": "center",
                },
            ],
        )

        cv2.imshow(WINDOW_NAME, frame)
        if window_closed(WINDOW_NAME):
            break
        to_launch = None
        launch_label = ""
        if clicked is not None:
            cx, cy = clicked
            for button in buttons:
                if point_in_button(cx, cy, button):
                    to_launch = button["script"]
                    launch_label = button["label"]
                    break
            ui_state["click_pos"] = None

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("f"):
            window_state["fullscreen"] = not window_state["fullscreen"]
            apply_window_mode(WINDOW_NAME, window_state["fullscreen"])
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)
        if key == ord("1"):
            to_launch = "freestyle.py"
            launch_label = "Freestyle"
        if key == ord("2"):
            to_launch = "lesson.py"
            launch_label = "Lesson"

        if to_launch:
            show_mode_loading(WINDOW_NAME, launch_label if launch_label else "Mode")
            run_mode_in_process(BASE_DIR, to_launch, WINDOW_NAME)
            apply_window_mode(WINDOW_NAME, window_state["fullscreen"])
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

finally:
    cv2.destroyAllWindows()




























