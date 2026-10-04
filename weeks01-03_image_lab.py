"""Weeks 1-3 Lab: Image Pixels, Filters, and Edges.

Run from the repository root:
    python3 weeks01-03_image_lab.py

Reads images/original.jpg and writes three labeled montages to outputs/.
The script never opens a GUI window (Matplotlib uses the non-interactive Agg backend).
"""

import json
import numbers
import os

import matplotlib

matplotlib.use("Agg")  # headless: save figures to files, never open a window

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

IMAGE_PATH = os.path.join("images", "original.jpg")
OUTPUT_DIR = "outputs"
BITS_PER_CHANNEL = 8


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_bgr_image(image_path: str) -> np.ndarray:
    """Load a colour image with OpenCV (returned in BGR channel order)."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'.")
    return image


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to a single-channel 8-bit grayscale image."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Reorder channels so Matplotlib displays colours correctly."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_montage(panels: list, output_path: str, title: str, columns: int) -> str:
    """Save a grid of labeled image panels.

    Each panel is a (label, image, cmap) tuple. cmap is None for colour (RGB)
    images and "gray" for single-channel images.
    """
    rows = int(np.ceil(len(panels) / columns))
    first_height, first_width = panels[0][1].shape[:2]
    panel_width = 5.0
    panel_height = panel_width * first_height / first_width + 0.6  # room for the label
    fig, axes = plt.subplots(rows, columns, figsize=(panel_width * columns, panel_height * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, (label, image, cmap) in zip(axes, panels):
        if cmap == "gray":
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            axis.imshow(image)
        axis.set_title(label, fontsize=11)
        axis.axis("off")

    for axis in axes[len(panels):]:
        axis.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Task 1 - Image data
# ---------------------------------------------------------------------------
def inspect_image(image_path: str) -> dict:
    """Load the image and return its measured image-data properties."""
    image = load_bgr_image(image_path)
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    pixel_count = width * height
    estimated_bytes = pixel_count * channels * BITS_PER_CHANNEL // 8

    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "shape": [int(value) for value in image.shape],
        "pixel_count": int(pixel_count),
        "estimated_bytes": int(estimated_bytes),
        # cv2.imread always returns colour data in Blue-Green-Red order.
        "color_order": "BGR",
    }


# ---------------------------------------------------------------------------
# Task 2 - Colour and resolution
# ---------------------------------------------------------------------------
def downsample_half(image: np.ndarray) -> np.ndarray:
    """Resize an image to half its width and height."""
    height, width = image.shape[:2]
    new_size = (max(1, width // 2), max(1, height // 2))  # cv2 uses (width, height)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Create the labeled channel, grayscale, and downsampled views."""
    os.makedirs(output_dir, exist_ok=True)
    image = load_bgr_image(image_path)
    height, width = image.shape[:2]

    blue, green, red = cv2.split(image)  # OpenCV order is B, G, R
    gray = to_grayscale(image)
    small = downsample_half(image)
    small_height, small_width = small.shape[:2]

    panels = [
        (f"Original (RGB view) {width}x{height}", bgr_to_rgb(image), None),
        ("Red channel (R intensity)", red, "gray"),
        ("Green channel (G intensity)", green, "gray"),
        ("Blue channel (B intensity)", blue, "gray"),
        (f"Grayscale {width}x{height}", gray, "gray"),
        (f"Downsampled 50% {small_width}x{small_height}", bgr_to_rgb(small), None),
    ]
    output_path = os.path.join(output_dir, "pixel_views.png")
    save_montage(panels, output_path, "Task 2 - Colour channels and resolution", columns=3)

    return {
        "original_size": [int(width), int(height)],
        "downsampled_size": [int(small_width), int(small_height)],
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Task 3 - Adjustments
# ---------------------------------------------------------------------------
def adjust_brightness(gray: np.ndarray, brightness_delta: int) -> np.ndarray:
    """Add a constant to every pixel and clip to the valid 0-255 range."""
    widened = gray.astype(np.int16) + int(brightness_delta)  # avoid uint8 wrap-around
    return np.clip(widened, 0, 255).astype(np.uint8)


def adjust_contrast(gray: np.ndarray, contrast_factor: float) -> np.ndarray:
    """Multiply every pixel by a factor and clip to the valid 0-255 range."""
    scaled = gray.astype(np.float32) * float(contrast_factor)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_threshold(gray: np.ndarray, threshold: int) -> np.ndarray:
    """Pixels greater than threshold become white (255); all others black (0)."""
    return np.where(gray > threshold, 255, 0).astype(np.uint8)


def validate_threshold(threshold) -> int:
    """Require an integer threshold in the range 0-255."""
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Integral):
        raise ValueError("threshold must be an integer in the range 0-255.")
    if not 0 <= threshold <= 255:
        raise ValueError(f"threshold must be in the range 0-255, got {threshold}.")
    return int(threshold)


def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Create labeled brightness, contrast, and threshold results."""
    threshold = validate_threshold(threshold)
    os.makedirs(output_dir, exist_ok=True)

    gray = to_grayscale(load_bgr_image(image_path))
    brighter = adjust_brightness(gray, brightness_delta)
    higher_contrast = adjust_contrast(gray, contrast_factor)
    black_white = apply_threshold(gray, threshold)

    panels = [
        ("Grayscale (input)", gray, "gray"),
        (f"Brighter (+{brightness_delta}, clipped)", brighter, "gray"),
        (f"Higher contrast (x{contrast_factor}, clipped)", higher_contrast, "gray"),
        (f"Threshold (> {threshold} = white)", black_white, "gray"),
    ]
    output_path = os.path.join(output_dir, "adjustments.png")
    save_montage(panels, output_path, "Task 3 - Grayscale adjustments", columns=2)

    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Task 4 - Blur and edges
# ---------------------------------------------------------------------------
def validate_kernel_size(kernel_size) -> int:
    """Require a positive odd integer kernel size."""
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, numbers.Integral):
        raise ValueError("kernel_size must be a positive odd integer.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
    return int(kernel_size)


def mean_blur(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Average each pixel with its kernel_size x kernel_size neighbourhood."""
    return cv2.blur(gray, (kernel_size, kernel_size))


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Return the Sobel gradient magnitude (float) from x and y derivatives."""
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(grad_x, grad_y)


def scale_to_uint8(magnitude: np.ndarray, reference_max: float) -> np.ndarray:
    """Scale edge strengths to 0-255 using a shared reference maximum."""
    if reference_max <= 0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    return np.clip(magnitude / reference_max * 255.0, 0, 255).astype(np.uint8)


def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Create labeled grayscale, mean-blur, and Sobel-edge results."""
    kernel_size = validate_kernel_size(kernel_size)
    os.makedirs(output_dir, exist_ok=True)

    gray = to_grayscale(load_bgr_image(image_path))
    blurred = mean_blur(gray, kernel_size)

    edges_original = sobel_magnitude(gray)
    edges_blurred = sobel_magnitude(blurred)
    # Use the same scale for both so edge strength can be compared fairly.
    # The 99.5th percentile ignores a few extreme pixels so ordinary edges stay visible.
    reference_max = float(np.percentile(edges_original, 99.5))
    edges_original_view = scale_to_uint8(edges_original, reference_max)
    edges_blurred_view = scale_to_uint8(edges_blurred, reference_max)

    # Close-up of the centre of the image, where blur differences are easier to see.
    height, width = gray.shape
    y0, y1 = int(height * 0.38), int(height * 0.73)
    x0, x1 = int(width * 0.25), int(width * 0.60)

    panels = [
        ("Grayscale", gray, "gray"),
        (f"Mean blur ({kernel_size}x{kernel_size})", blurred, "gray"),
        ("Sobel edges - original grayscale", edges_original_view, "gray"),
        ("Sobel edges - blurred grayscale", edges_blurred_view, "gray"),
        ("Close-up: edges - original", edges_original_view[y0:y1, x0:x1], "gray"),
        ("Close-up: edges - blurred", edges_blurred_view[y0:y1, x0:x1], "gray"),
    ]
    output_path = os.path.join(output_dir, "blur_and_edges.png")
    save_montage(panels, output_path, "Task 4 - Mean blur and Sobel edges", columns=2)

    return {
        "kernel_size": kernel_size,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Task 5 - Reproducible run
# ---------------------------------------------------------------------------
def run_lab(image_path: str, output_dir: str) -> dict:
    """Run Tasks 1-4 and return their results together."""
    os.makedirs(output_dir, exist_ok=True)
    return {
        "task1": inspect_image(image_path),
        "task2": create_pixel_views(image_path, output_dir),
        "task3": create_adjustments(image_path, output_dir),
        "task4": create_blur_and_edges(image_path, output_dir),
    }


def main() -> None:
    """Run the lab using the required repository paths."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = run_lab(IMAGE_PATH, OUTPUT_DIR)

    info = results["task1"]
    print("Task 1 - Image data")
    print(f"  Width x height : {info['width']} x {info['height']}")
    print(f"  Channels       : {info['channels']}")
    print(f"  NumPy shape    : {info['shape']}")
    print(f"  Pixel count    : {info['pixel_count']:,}")
    print(f"  Estimated size : {info['estimated_bytes']:,} bytes (8 bits per channel)")
    print(f"  Colour order   : {info['color_order']}")
    print()
    print("Full results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
