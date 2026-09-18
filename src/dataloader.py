from imports import *

from augmentations import apply_random_augmentation, current_augmentation_strength
from structural_prior_and_boundary_generation import generate_guiding_map

# =========================
# Add guidance to ImageLabelGenerator Alongwith Augmnetation Logic
# =========================

class ImageLabelGenerator(Sequence):

    def __init__(
        self,
        image_label_pairs,
        batch_size=2,
        target_size=(512, 704),
        augment=False,
        shuffle=True,
        use_guidance=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_label_pairs = image_label_pairs
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.use_guidance = use_guidance

        self.indexes = np.arange(len(self.image_label_pairs))
        self.image_filenames = [
            img_path for img_path, _ in self.image_label_pairs
        ]

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(
            len(self.image_label_pairs) / self.batch_size
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_items = [
            self.image_label_pairs[i]
            for i in self.indexes[
                idx * self.batch_size:(idx + 1) * self.batch_size
            ]
        ]

        images, labels = [], []

        for img_path, lbl_path in batch_items:
            try:
                if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                    print(f"[SKIP] Missing file: {img_path} or {lbl_path}")
                    continue

                # Image loading.
                img_ext = os.path.splitext(img_path)[-1].lower()
                img = None

                if img_ext in [".tif", ".tiff"]:
                    img = tiff.imread(img_path)

                    # Some broken TIFFs may return None or empty data.
                    if img is None or img.size == 0:
                        print(f"[ERROR] Empty TIFF: {img_path}")
                        continue

                    if img.ndim == 2:
                        img = np.stack([img] * 3, axis=-1)

                    elif img.ndim == 3:
                        if img.shape[0] in [1, 2, 3]:
                            img = np.transpose(img, (1, 2, 0))

                        if img.shape[-1] > 3:
                            img = img[:, :, :3]

                else:
                    img = load_img(
                        img_path,
                        target_size=self.target_size,
                        color_mode="rgb"
                    )
                    img = img_to_array(img)

                # Safety check before resize.
                if img is None or img.size == 0:
                    print(f"[ERROR] Invalid image data at {img_path}")
                    continue

                img = img.astype(np.float32) / 255.0

                img = cv2.resize(
                    img,
                    self.target_size[::-1],
                    interpolation=cv2.INTER_LINEAR
                )

                # Input guidance.
                if self.use_guidance:
                    guiding_map = generate_guiding_map(img)
                    img = np.concatenate(
                        [img, guiding_map],
                        axis=-1
                    )

                # Label loading.
                lbl = tiff.imread(lbl_path).astype(np.float32)

                if lbl is None or lbl.size == 0:
                    print(f"[ERROR] Empty label TIFF: {lbl_path}")
                    continue

                if lbl.ndim == 3:
                    if lbl.shape[0] in [1, 2, 3]:
                        lbl = lbl[0]
                    elif lbl.shape[-1] == 1:
                        lbl = lbl[..., 0]

                lbl = lbl / (lbl.max() if lbl.max() > 1 else 1.0)

                lbl = cv2.resize(
                    lbl,
                    self.target_size[::-1],
                    interpolation=cv2.INTER_NEAREST
                )

                # Check for mismatched shapes.
                if img.shape[:2] != lbl.shape[:2]:
                    print(
                        f"[WARN] Mismatch: {os.path.basename(img_path)} "
                        f"-> img={img.shape}, lbl={lbl.shape}"
                    )

                    lbl = cv2.resize(
                        lbl,
                        img.shape[:2][::-1],
                        interpolation=cv2.INTER_NEAREST
                    )

                # Apply curriculum-based augmentation.
                if self.augment:
                    # Apply augmentation to image.
                    aug_img = apply_random_augmentation(img)

                    # For label, apply only geometric transforms.
                    if np.random.rand() < current_augmentation_strength:
                        k = np.random.choice([0, 1, 2, 3])

                        lbl = np.rot90(lbl, k)
                        aug_img = np.rot90(aug_img, k)

                        if np.random.rand() < 0.5:
                            aug_img = np.flip(aug_img, axis=1)
                            lbl = np.flip(lbl, axis=1)

                    img = cv2.resize(
                        aug_img,
                        self.target_size[::-1],
                        interpolation=cv2.INTER_LINEAR
                    )

                    lbl = cv2.resize(
                        lbl,
                        self.target_size[::-1],
                        interpolation=cv2.INTER_NEAREST
                    )

                if lbl.ndim == 2:
                    lbl = np.expand_dims(lbl, axis=-1)

                images.append(img)
                labels.append(lbl)

            except Exception as e:
                print(
                    f"[ERROR] Failed to load: "
                    f"{img_path}, {lbl_path} -> {e}"
                )
                continue

        if len(images) == 0:
            dummy_image = np.zeros(
                (*self.target_size, 3 + int(self.use_guidance)),
                dtype=np.float32
            )

            dummy_label = np.zeros(
                (*self.target_size, 1),
                dtype=np.float32
            )

            return np.array([dummy_image]), np.array([dummy_label])

        # Validate all shapes before converting to np.array.
        img_shapes = [im.shape for im in images]
        lbl_shapes = [lb.shape for lb in labels]

        if len(set(img_shapes)) > 1 or len(set(lbl_shapes)) > 1:
            print("[ERROR] Inconsistent shapes in batch:")
            print("Images:", img_shapes)
            print("Labels:", lbl_shapes)

            # Remove problematic samples.
            min_shape = images[0].shape

            images = [
                im for im in images
                if im.shape == min_shape
            ]

            labels = [
                lb for lb in labels
                if lb.shape == (min_shape[0], min_shape[1], 1)
            ]

        return (
            np.array(images, dtype=np.float32),
            np.array(labels, dtype=np.float32)
        )


# ===============================
# Utility: Get image-label pairs
# ===============================

def get_image_label_pairs(image_dir, label_dir):
    valid_exts = (
        ".tiff",
        ".tif",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp"
    )

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(valid_exts)
    ]

    pairs = []

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)

        # Match any label file with the same base name.
        for ext in valid_exts:
            lbl_candidate = os.path.join(
                label_dir,
                base_name + ext
            )

            if os.path.exists(lbl_candidate):
                pairs.append((img_path, lbl_candidate))
                break

    return pairs
