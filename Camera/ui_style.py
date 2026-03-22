import os
import re

# Shared style loader for ManuMano UI themes (reads style.css variables).

_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_VAR_RE = re.compile(r"--([a-zA-Z0-9_-]+)\s*:\s*([^;]+);")


def load_style_vars(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "style.css")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return {}

    raw = _COMMENT_RE.sub("", raw)

    data = {}
    for key, value in _VAR_RE.findall(raw):
        clean = value.strip().strip('"').strip("'")
        data[key.strip().lower()] = clean

    return data


def _parse_hex_color(value):
    token = value.strip().lstrip("#")
    if len(token) == 3:
        token = "".join(ch * 2 for ch in token)

    if len(token) != 6:
        return None

    try:
        r = int(token[0:2], 16)
        g = int(token[2:4], 16)
        b = int(token[4:6], 16)
    except ValueError:
        return None

    return (r, g, b)


def parse_color(value):
    if not value:
        return None

    token = value.strip().lower()
    if token.startswith("#"):
        return _parse_hex_color(token)

    if token.startswith("rgb(") and token.endswith(")"):
        body = token[4:-1]
        parts = [p.strip() for p in body.split(",")]
        if len(parts) != 3:
            return None

        try:
            r = int(float(parts[0]))
            g = int(float(parts[1]))
            b = int(float(parts[2]))
        except ValueError:
            return None

        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    return None


def palette_from_vars(style_vars, defaults):
    palette = {}
    for key, fallback in defaults.items():
        css_key = key.replace("_", "-")
        parsed = parse_color(style_vars.get(css_key, ""))
        palette[key] = parsed if parsed is not None else fallback
    return palette


def get_css_str(style_vars, key, default):
    value = style_vars.get(key.lower())
    return value if value else default


def get_css_int(style_vars, key, default):
    value = style_vars.get(key.lower())
    if not value:
        return default

    token = value.strip().lower()
    if token.endswith("px"):
        token = token[:-2]

    try:
        return int(float(token))
    except ValueError:
        return default


def get_css_float(style_vars, key, default):
    value = style_vars.get(key.lower())
    if not value:
        return default

    token = value.strip().lower()
    if token.endswith("px"):
        token = token[:-2]

    try:
        return float(token)
    except ValueError:
        return default


def resolve_font_candidates(style_vars, css_key, defaults):
    candidates = []

    custom = style_vars.get(css_key.lower(), "").strip()
    if custom:
        candidates.append(custom)
        if not os.path.isabs(custom):
            candidates.append(os.path.join(os.getcwd(), custom))
            candidates.append(os.path.join(os.path.dirname(__file__), custom))

    candidates.extend(defaults)

    deduped = []
    seen = set()
    for path in candidates:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(path)

    return deduped
