# ===============================
# Input-Level Guidance Map (Same as the heuristic generation pipeline of GPiSeT with minor modifications).
# ===============================

def generate_guiding_map(img):
    """
    Generate a multi-modality guidance map for cell segmentation.

    Supports RGB, grayscale, and multi-channel images.
    The guidance map is generated using:
    1. Grayscale conversion or max projection
    2. Adaptive contrast enhancement
    3. Gaussian blur
    4. Otsu thresholding
    5. Morphological opening and closing
    6. Min-max normalization
    """

    # Convert to grayscale if needed.
    if img.ndim == 3:
        if img.shape[2] == 3:
            img_gray = cv2.cvtColor(
                img.astype(np.float32),
                cv2.COLOR_RGB2GRAY
            )
        else:
            # Use max projection for multi-channel images.
            img_gray = img.max(axis=-1)
    else:
        img_gray = img.astype(np.float32)

    # Normalize and enhance contrast.
    img_gray = (
        (img_gray - img_gray.min()) /
        (img_gray.max() - img_gray.min() + 1e-8)
    )

    img_contrast = exposure.equalize_adapthist(
        img_gray,
        clip_limit=0.03
    )

    # Apply Gaussian blur.
    img_blur = cv2.GaussianBlur(
        img_contrast,
        (3, 3),
        0
    )

    # Apply Otsu thresholding.
    try:
        t = threshold_otsu(img_blur)
    except ValueError:
        t = 0.5

    mask = np.maximum(
        img_blur - t,
        0
    )

    # Apply morphological opening and closing.
    mask = opening(
        mask,
        footprint_rectangle((3, 3))
    )

    mask = closing(
        mask,
        footprint_rectangle((3, 3))
    )

    # Apply min-max normalization.
    mask_norm = (
        (mask - mask.min()) /
        (mask.max() - mask.min() + 1e-8)
    )

    return mask_norm[..., np.newaxis]


# ===============================
# Refined Sobel Edge Map
# ===============================

def refined_sobel_edges(image):
    """Compute a refined Sobel edge map."""
    gray = cv2.cvtColor(
        image.astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    grad = np.sqrt(gx**2 + gy**2)

    grad = cv2.normalize(
        grad,
        None,
        0,
        1,
        cv2.NORM_MINMAX
    )

    return grad
