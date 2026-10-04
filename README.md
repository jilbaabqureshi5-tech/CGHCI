# Weeks 1–3 Lab: Image Pixels, Filters, and Edges

This repository loads my own photograph (`images/original.jpg`), reports its image-data properties, and produces three labeled montages: colour channels and resolution, grayscale adjustments, and mean blur with Sobel edges.

## Setup

Requires Python 3.9 or newer.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python3 weeks01-03_image_lab.py
```

The script reads `images/original.jpg`, creates `outputs/` if needed, writes `outputs/pixel_views.png`, `outputs/adjustments.png` and `outputs/blur_and_edges.png`, and prints the Task 1 values. It runs without opening any window.

## Repository contents

| Path | Purpose |
|---|---|
| `images/original.jpg` | My own photograph, saved unchanged |
| `weeks01-03_image_lab.py` | Lab script (Tasks 1–5) |
| `outputs/pixel_views.png` | Task 2 montage |
| `outputs/adjustments.png` | Task 3 montage |
| `outputs/blur_and_edges.png` | Task 4 montage |
| `requirements.txt` | Dependencies |

## Task 1 — Image data

| Property | Value |
|---|---|
| Width | 3048 px |
| Height | 4064 px |
| Channels | 3 |
| NumPy shape | [4064, 3048, 3] |
| Pixel count | 12,387,072 |
| Estimated size (8 bits per channel) | 37,161,216 bytes (about 37.2 MB) |
| Colour order as loaded | BGR |

The NumPy array is ordered (rows, columns, channels): the first number is the height, the second is the width, and the third is the three colour values stored per pixel. Pixel count is width × height. Because each channel uses 8 bits (1 byte), the estimated size is width × height × 3 bytes. This is the uncompressed size in memory, so it is much larger than the JPEG file on disk (about 2.7 MB), because JPEG compression removes redundant information. The photo is in portrait orientation, so the height is greater than the width. OpenCV's `cv2.imread` stores the channels as Blue, Green, Red, so the script converts to RGB only when displaying colour images with Matplotlib.

## Task 2 — Colour and resolution

- Downsampled dimensions: 1524 × 2032 px (half of the original 3048 × 4064, using area interpolation). The half-size image has 3,096,768 pixels, a quarter of the original.
- Lost detail: the small printed numbers on the watch dial (such as "6.5" and "7.5") and the tiny minute markers become softer and blockier, and the fine scratches on the black bezel are much harder to see.
- Channel difference: the yellow notebook cover and the lime-green sticky note look bright in the red and green channels but dark in the blue channel. Yellow is made of red and green light with very little blue (the notebook averages about R 156, G 108, B 36). The green folder is brighter in the green channel than in the red channel, while the black watch is dark in all three channels.

## Task 3 — Adjustments

All adjustments start from the grayscale image and are implemented in named functions (`adjust_brightness`, `adjust_contrast`, `apply_threshold`).

| Adjustment | Setting | Operation |
|---|---|---|
| Brightness | `brightness_delta = 40` | Adds 40 to every pixel, then clips to 0–255 |
| Contrast | `contrast_factor = 1.5` | Multiplies every pixel by 1.5, then clips to 0–255 |
| Threshold | `threshold = 127` | Pixels **greater than** 127 become white (255); all others become black (0) |

Values are calculated in a wider data type before clipping, so bright pixels saturate at 255 instead of wrapping around to dark values. The threshold must be an integer from 0 to 255, otherwise the function raises `ValueError`.

Observations: the grayscale image has an average value of about 116. Brightening by 40 lifts the whole image evenly, and only about 0.5% of pixels (the brightest parts of the white folder) are clipped to pure white. Multiplying by 1.5 pushes about 7.7% of pixels to white, so the white folder and parts of the sticky note lose detail, while the black watch gets only slightly lighter, which increases the difference between light and dark areas. With a threshold of 127, about 44% of pixels become white: the sticky note and bright parts of the notebook turn white, while the watch, its strap and the spiral binding turn black. The handwriting on the lower part of the card remains readable, but the top line ("HCCGI Image") is partly lost because the watch's shadow makes that part of the card darker than 127.

## Task 4 — Blur and edges

- Mean blur: 5 × 5 kernel (`cv2.blur`). `kernel_size` must be a positive odd integer.
- Sobel edges: horizontal and vertical gradients (`cv2.Sobel`, 3 × 3) combined as gradient magnitude. Both edge images use the same brightness scale (based on the 99.5th percentile of the original edge strengths) so they can be compared fairly.

The montage also includes a close-up of the watch and the card text, because the 5 × 5 blur is small compared with a 3048 × 4064 image and its effect is easier to see when enlarged.

Edge comparison: the outline of the handwritten letters on the sticky note appears as a thin, sharp double line in the original edge image. After blurring, the same strokes become wider, smoother, single bands that are slightly fainter. The lower border of the sticky note shows the same change: a crisp thin line becomes a softer, wider one. Overall, the average edge strength drops from about 10.4 to 6.6 after blurring.

Detail reduced by blur: the fine diamond texture on the watch dial and the tiny printed text ("WATER RESISTANT 30M") almost disappear in the blurred edge image, and the speckled noise from the paper and plastic surfaces is mostly removed.

## Summary of observations

The photo is stored as a 4064 × 3048 × 3 array of 8-bit values in BGR order, which needs about 37 MB in memory without compression. Splitting the channels showed that colours are combinations of red, green and blue intensity: yellow objects are bright in red and green but dark in blue. Halving the resolution kept the overall scene but removed the smallest details, such as the numbers on the watch dial. Brightness and contrast changes must be clipped to 0–255, and contrast causes much more clipping than brightness in this photo. Thresholding depends strongly on lighting, since a shadow was enough to turn part of the card black. Mean blur reduced noise and fine texture, which made the Sobel edges cleaner but weaker and wider.

## Sources

- OpenCV documentation: https://docs.opencv.org/4.x/
- NumPy documentation: https://numpy.org/doc/
- Matplotlib documentation: https://matplotlib.org/stable/
