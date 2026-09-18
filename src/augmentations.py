def zoom_image(img, scale_range=(0.85, 1.15)):
    """Apply mild zoom-in or zoom-out without severe cropping or padding artifacts."""
    scale = np.random.uniform(*scale_range)
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    img_zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad or crop back to the original size.
    if scale < 1.0:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        img_padded = np.zeros_like(img)
        img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_zoomed
        return img_padded
    else:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return img_zoomed[start_h:start_h+h, start_w:start_w+w]


def rotate_image(img, k=None):
    """Apply a 90-degree rotation."""
    if k is None:
        k = np.random.choice([0, 1, 2, 3])
    return np.rot90(img, k=k)


def cell_aware_intensity_scale(img, scale_range=(0.9, 1.2)):
    """Apply gentle brightness scaling."""
    return np.clip(img * np.random.uniform(*scale_range), 0, 1)


def add_gaussian_noise(img, mean=0, std=0.02):
    """Add low-level Gaussian noise without destroying image structure."""
    noise = np.random.normal(mean, std, img.shape)
    return np.clip(img + noise, 0, 1)


def contrast_adjustment(img, gamma_range=(0.8, 1.2)):
    """Apply subtle gamma-based contrast adjustment."""
    gamma = np.random.uniform(*gamma_range)
    return np.clip(np.power(img, gamma), 0, 1)


def gaussian_smoothing(img, sigma=0.5):
    """Apply mild Gaussian smoothing."""
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# Curriculum learning control variable.
current_augmentation_strength = 0.0  # Starts with easier augmentations.


def apply_random_augmentation(img):
    """
    Apply random augmentation based on the global curriculum strength.
    As training progresses, augmentations become more frequent and stronger.
    """
    global current_augmentation_strength

    aug_funcs = [
        lambda x: zoom_image(
            x,
            scale_range=(
                0.95 - 0.1 * current_augmentation_strength,
                1.05 + 0.1 * current_augmentation_strength
            )
        ),
        lambda x: rotate_image(x, k=np.random.choice([0, 1, 2, 3])),
        lambda x: cell_aware_intensity_scale(
            x,
            scale_range=(
                1 - 0.2 * current_augmentation_strength,
                1 + 0.2 * current_augmentation_strength
            )
        ),
        lambda x: add_gaussian_noise(
            x,
            std=0.01 + 0.03 * current_augmentation_strength
        ),
        lambda x: contrast_adjustment(
            x,
            gamma_range=(
                0.9 - 0.1 * current_augmentation_strength,
                1.1 + 0.1 * current_augmentation_strength
            )
        ),
        lambda x: gaussian_smoothing(
            x,
            sigma=0.3 + 0.5 * current_augmentation_strength
        )
    ]

    # Increase augmentation probability with curriculum strength.
    prob = 0.2 + 0.6 * current_augmentation_strength  # 0.2 -> 0.8

    if np.random.rand() < prob:
        aug = np.random.choice(aug_funcs)
        return aug(img)

    return img
