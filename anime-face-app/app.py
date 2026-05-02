from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

APP_NAME = "AnimeForge"
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "12"))
LOGGER = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    path: Path
    engine: str


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    _ensure_directories()

    @app.route("/")
    def home() -> str:
        return render_template("home.html", app_name=APP_NAME)

    @app.route("/text-to-anime", methods=["GET", "POST"])
    def text_to_anime() -> str:
        prompt = ""
        result: GenerationResult | None = None

        if request.method == "POST":
            prompt = request.form.get("prompt", "").strip()
            if len(prompt) < 3:
                flash("Veuillez saisir un prompt plus descriptif (au moins 3 caracteres).", "warning")
            else:
                try:
                    result = generate_text_to_anime(prompt)
                    flash(f"Image generee via {result.engine}.", "success")
                except Exception as exc:  # pragma: no cover - robust error path
                    app.logger.exception("Text-to-anime failed: %s", exc)
                    flash("La generation a echoue. Verifiez vos modeles ou reessayez.", "danger")

        return render_template(
            "text_to_anime.html",
            app_name=APP_NAME,
            prompt=prompt,
            result=result,
        )

    @app.route("/random-anime", methods=["GET", "POST"])
    def random_anime() -> str:
        result: GenerationResult | None = None

        if request.method == "POST":
            try:
                result = generate_random_anime()
                flash(f"Personnage genere via {result.engine}.", "success")
            except Exception as exc:  # pragma: no cover - robust error path
                app.logger.exception("Random anime failed: %s", exc)
                flash("La generation aleatoire a echoue. Reessayez.", "danger")

        return render_template("random_anime.html", app_name=APP_NAME, result=result)

    @app.route("/face-to-anime", methods=["GET", "POST"])
    def face_to_anime() -> str:
        result: GenerationResult | None = None
        uploaded_filename: str | None = None

        if request.method == "POST":
            file = request.files.get("face_image")
            if file is None or not file.filename:
                flash("Veuillez selectionner une image visage a transformer.", "warning")
            elif not _allowed_file(file.filename):
                flash("Format non supporte. Utilisez PNG, JPG, JPEG ou WEBP.", "warning")
            else:
                try:
                    upload_path = _save_uploaded_file(file)
                    uploaded_filename = upload_path.name
                    result = convert_face_to_anime(upload_path)
                    flash(f"Transformation terminee via {result.engine}.", "success")
                except Exception as exc:  # pragma: no cover - robust error path
                    app.logger.exception("Face-to-anime failed: %s", exc)
                    flash("La transformation a echoue. Verifiez le modele puis reessayez.", "danger")

        return render_template(
            "face_to_anime.html",
            app_name=APP_NAME,
            result=result,
            uploaded_filename=uploaded_filename,
        )

    @app.route("/outputs/<path:filename>")
    def output_file(filename: str):
        return send_from_directory(OUTPUT_DIR, filename)

    @app.route("/uploads/<path:filename>")
    def upload_file(filename: str):
        return send_from_directory(UPLOAD_DIR, filename)

    @app.route("/download/<path:filename>")
    def download_output(filename: str):
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_file(_error):
        flash(f"Fichier trop volumineux. Limite: {MAX_UPLOAD_MB} MB.", "danger")
        return redirect(request.referrer or url_for("home"))

    @app.context_processor
    def inject_globals():
        return {"year": datetime.now(UTC).year}

    return app


def _ensure_directories() -> None:
    for path in (
        MODELS_DIR,
        MODELS_DIR / "text-to-anime",
        MODELS_DIR / "random-anime",
        MODELS_DIR / "human-to-anime",
        UPLOAD_DIR,
        OUTPUT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _build_filename(prefix: str, extension: str = ".png") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}{extension}"


def _save_uploaded_file(file_storage) -> Path:
    ext = Path(secure_filename(file_storage.filename)).suffix.lower() or ".png"
    filename = _build_filename("upload", ext)
    destination = UPLOAD_DIR / filename
    file_storage.save(destination)
    return destination


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _extract_prompt_colors(prompt: str, rng: random.Random) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    text = prompt.lower()

    hair_map = {
        "blue": (67, 129, 255),
        "bleu": (67, 129, 255),
        "red": (255, 87, 87),
        "rouge": (255, 87, 87),
        "pink": (255, 127, 191),
        "rose": (255, 127, 191),
        "green": (71, 207, 152),
        "vert": (71, 207, 152),
        "black": (39, 46, 72),
        "noir": (39, 46, 72),
        "blond": (255, 214, 109),
        "yellow": (255, 214, 109),
        "violet": (160, 122, 255),
        "purple": (160, 122, 255),
        "white": (234, 241, 255),
        "blanc": (234, 241, 255),
        "brown": (140, 98, 74),
        "brun": (140, 98, 74),
    }
    eye_map = {
        "blue": (87, 176, 255),
        "bleu": (87, 176, 255),
        "green": (90, 222, 162),
        "vert": (90, 222, 162),
        "red": (255, 108, 113),
        "rouge": (255, 108, 113),
        "gold": (247, 210, 112),
        "amber": (247, 210, 112),
        "violet": (175, 139, 255),
    }

    hair_color = next((value for key, value in hair_map.items() if key in text), None)
    eye_color = next((value for key, value in eye_map.items() if key in text), None)

    if hair_color is None:
        hair_color = rng.choice(list(hair_map.values()))
    if eye_color is None:
        eye_color = rng.choice(list(eye_map.values()))

    skin_palette = [
        (255, 222, 199),
        (247, 208, 176),
        (237, 191, 158),
        (226, 176, 142),
        (206, 156, 125),
        (184, 134, 106),
    ]
    skin_color = rng.choice(skin_palette)
    return hair_color, eye_color, skin_color


def _draw_gradient_background(draw: ImageDraw.ImageDraw, width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    for y in range(height):
        ratio = y / max(1, height - 1)
        color = (
            int(top[0] * (1.0 - ratio) + bottom[0] * ratio),
            int(top[1] * (1.0 - ratio) + bottom[1] * ratio),
            int(top[2] * (1.0 - ratio) + bottom[2] * ratio),
        )
        draw.line((0, y, width, y), fill=color)


def _render_synthetic_anime(output_path: Path, *, prompt: str, seed: int) -> None:
    rng = random.Random(seed)
    width, height = 768, 768

    top = (8, 16, 38)
    bottom = (20, 48, 74)
    canvas = Image.new("RGB", (width, height), top)
    draw = ImageDraw.Draw(canvas, "RGBA")

    _draw_gradient_background(draw, width, height, top, bottom)

    for _ in range(14):
        glow_radius = rng.randint(60, 160)
        glow_x = rng.randint(-80, width + 80)
        glow_y = rng.randint(-80, height + 80)
        glow_color = rng.choice(
            [
                (42, 248, 213, 48),
                (73, 184, 255, 54),
                (255, 151, 102, 46),
                (202, 127, 255, 42),
            ]
        )
        draw.ellipse(
            (glow_x - glow_radius, glow_y - glow_radius, glow_x + glow_radius, glow_y + glow_radius),
            fill=glow_color,
        )

    hair_color, eye_color, skin_color = _extract_prompt_colors(prompt, rng)

    shoulder_y = int(height * 0.72)
    draw.ellipse((-120, shoulder_y - 100, width + 120, height + 200), fill=(37, 52, 84, 220))

    neck_w = 120
    neck_h = 130
    neck_x = width // 2
    draw.rounded_rectangle(
        (neck_x - neck_w // 2, shoulder_y - 80, neck_x + neck_w // 2, shoulder_y - 80 + neck_h),
        radius=26,
        fill=skin_color + (255,),
    )

    head_w = 330
    head_h = 390
    head_left = width // 2 - head_w // 2
    head_top = 160
    head_right = head_left + head_w
    head_bottom = head_top + head_h

    draw.ellipse((head_left, head_top, head_right, head_bottom), fill=skin_color + (255,))

    hair_shadow = tuple(max(channel - 24, 0) for channel in hair_color)
    draw.polygon(
        [
            (head_left - 40, head_top + 70),
            (width // 2, head_top - 78),
            (head_right + 48, head_top + 66),
            (head_right + 15, head_top + 280),
            (head_left - 15, head_top + 280),
        ],
        fill=hair_shadow + (255,),
    )
    draw.polygon(
        [
            (head_left - 20, head_top + 82),
            (width // 2, head_top - 58),
            (head_right + 25, head_top + 88),
            (head_right - 5, head_top + 244),
            (head_left + 6, head_top + 246),
        ],
        fill=hair_color + (255,),
    )

    fringe_points = []
    points_count = 7
    for index in range(points_count):
        ratio = index / (points_count - 1)
        x = int(head_left + 26 + ratio * (head_w - 52))
        y = int(head_top + 70 + rng.randint(-6, 28) + (index % 2) * 28)
        fringe_points.append((x, y))
    fringe_points.extend([(head_right - 20, head_top + 40), (head_left + 20, head_top + 44)])
    draw.polygon(fringe_points, fill=tuple(min(channel + 18, 255) for channel in hair_color) + (255,))

    eye_y = head_top + 175
    eye_w, eye_h = 76, 45
    left_eye_x = width // 2 - 100
    right_eye_x = width // 2 + 24

    for eye_x in (left_eye_x, right_eye_x):
        draw.ellipse((eye_x, eye_y, eye_x + eye_w, eye_y + eye_h), fill=(246, 251, 255, 255))
        iris_box = (eye_x + 16, eye_y + 9, eye_x + eye_w - 16, eye_y + eye_h + 10)
        draw.ellipse(iris_box, fill=eye_color + (255,))
        pupil_box = (eye_x + 30, eye_y + 20, eye_x + eye_w - 30, eye_y + eye_h + 8)
        draw.ellipse(pupil_box, fill=(17, 28, 42, 255))
        draw.ellipse((eye_x + 24, eye_y + 14, eye_x + 34, eye_y + 24), fill=(255, 255, 255, 220))

    brow_color = tuple(max(channel - 38, 0) for channel in hair_color)
    draw.line((left_eye_x - 2, eye_y - 14, left_eye_x + eye_w + 4, eye_y - 18), fill=brow_color + (220,), width=6)
    draw.line((right_eye_x - 6, eye_y - 18, right_eye_x + eye_w + 2, eye_y - 14), fill=brow_color + (220,), width=6)

    nose_x = width // 2
    nose_y = head_top + 236
    draw.line((nose_x, nose_y - 8, nose_x - 4, nose_y + 16), fill=(190, 137, 120, 130), width=3)

    mouth_top = head_top + 284
    smile_intensity = 16 if "smile" in prompt.lower() or "sourire" in prompt.lower() else 8
    draw.arc((width // 2 - 46, mouth_top, width // 2 + 46, mouth_top + 30 + smile_intensity), start=8, end=172, fill=(179, 76, 98, 220), width=4)

    blush = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    blush_draw = ImageDraw.Draw(blush, "RGBA")
    blush_draw.ellipse((left_eye_x - 22, eye_y + 52, left_eye_x + 66, eye_y + 102), fill=(255, 138, 153, 72))
    blush_draw.ellipse((right_eye_x + 10, eye_y + 54, right_eye_x + 98, eye_y + 104), fill=(255, 138, 153, 72))
    blush = blush.filter(ImageFilter.GaussianBlur(9))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), blush).convert("RGB")

    if "cat" in prompt.lower() or "neko" in prompt.lower():
        ear_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ear_draw = ImageDraw.Draw(ear_layer, "RGBA")
        left_ear = [(head_left + 40, head_top + 34), (head_left + 86, head_top - 90), (head_left + 146, head_top + 20)]
        right_ear = [(head_right - 146, head_top + 20), (head_right - 86, head_top - 90), (head_right - 40, head_top + 34)]
        ear_draw.polygon(left_ear, fill=hair_color + (255,))
        ear_draw.polygon(right_ear, fill=hair_color + (255,))
        ear_draw.polygon([(head_left + 76, head_top + 14), (head_left + 90, head_top - 56), (head_left + 124, head_top + 16)], fill=(255, 174, 186, 220))
        ear_draw.polygon([(head_right - 124, head_top + 16), (head_right - 90, head_top - 56), (head_right - 76, head_top + 14)], fill=(255, 174, 186, 220))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), ear_layer).convert("RGB")

    canvas = ImageEnhance.Color(canvas).enhance(1.15)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.08)
    canvas = canvas.filter(ImageFilter.SMOOTH_MORE)
    canvas.save(output_path)


def _stylize_face_fallback(image_path: Path, output_path: Path) -> None:
    image = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    base = image.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.SMOOTH_MORE)
    base = ImageOps.posterize(base, 4)
    base = ImageEnhance.Color(base).enhance(1.6)
    base = ImageEnhance.Contrast(base).enhance(1.18)

    edges = image.filter(ImageFilter.FIND_EDGES).convert("L")
    edges = ImageEnhance.Contrast(edges).enhance(2.0)
    edges = ImageOps.invert(edges)
    edges = ImageOps.colorize(edges, black=(10, 16, 28), white=(255, 255, 255))
    anime = ImageChops.multiply(base, edges)

    anime = ImageEnhance.Sharpness(anime).enhance(1.35)
    anime = anime.filter(ImageFilter.SMOOTH)
    anime.save(output_path)


def _load_hook(module_path: Path, function_name: str) -> Callable | None:
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(f"anime_hook_{module_path.stem}_{uuid4().hex[:8]}", module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name, None)


def _try_custom_text_model(prompt: str, output_path: Path) -> str | None:
    hook = _load_hook(MODELS_DIR / "text-to-anime" / "inference.py", "generate")
    if hook is None:
        return None
    try:
        hook(prompt, str(output_path))
    except Exception:
        return None
    if output_path.exists():
        return "Custom Text Model"
    return None


def _try_custom_random_model(output_path: Path) -> str | None:
    hook = _load_hook(MODELS_DIR / "random-anime" / "inference.py", "generate")
    if hook is None:
        return None
    try:
        hook(str(output_path))
    except Exception:
        return None
    if output_path.exists():
        return "Custom Random Model"
    return None


def _try_custom_face_model(image_path: Path, output_path: Path) -> str | None:
    hook = _load_hook(MODELS_DIR / "human-to-anime" / "inference.py", "transform")
    if hook is None:
        LOGGER.warning("No face-to-anime hook found at models/human-to-anime/inference.py")
        return None
    try:
        hook(str(image_path), str(output_path))
        if output_path.exists():
            return "U-GAT-IT (models/human-to-anime/best.pt)"
    except Exception as exc:
        LOGGER.exception("Face-to-anime custom hook failed: %s", exc)
    return None


def generate_text_to_anime(prompt: str) -> GenerationResult:
    output_path = OUTPUT_DIR / _build_filename("text_to_anime")

    engine = _try_custom_text_model(prompt, output_path)
    if engine is None:
        seed = _seed_from_text(prompt)
        _render_synthetic_anime(output_path, prompt=prompt, seed=seed)
        engine = "Prompt-to-style fallback"

    return GenerationResult(path=output_path, engine=engine)


def generate_random_anime() -> GenerationResult:
    output_path = OUTPUT_DIR / _build_filename("random_anime")

    engine = _try_custom_random_model(output_path)
    if engine is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
        prompt_bank = [
            "heroine cyberpunk avec cheveux bleus",
            "samurai anime avec armure lumineuse",
            "idol futuriste avec regard vert",
            "mage celeste style anime",
            "personnage neko avec cheveux roses",
        ]
        prompt = random.choice(prompt_bank)
        _render_synthetic_anime(output_path, prompt=prompt, seed=seed)
        engine = "Random style fallback"

    return GenerationResult(path=output_path, engine=engine)


def convert_face_to_anime(image_path: Path) -> GenerationResult:
    output_path = OUTPUT_DIR / _build_filename("face_to_anime")

    engine = _try_custom_face_model(image_path, output_path)
    if engine is None:
        _stylize_face_fallback(image_path, output_path)
        engine = "Classic stylizer fallback"

    return GenerationResult(path=output_path, engine=engine)


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
