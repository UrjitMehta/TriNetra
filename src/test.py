from imports import *

from dataloader import ImageLabelGenerator
from metrics import bce_dice_loss, dice_coef, F1Score, ap_metric, AFNR
from model import swin_unet
from structural_prior_and_boundary_generation import generate_guiding_map
# ============================================================
# Test Evaluation with COCO and Visualizations
# ============================================================


# ============================================================
# TIFF Helper
# ============================================================

def tiff_imread(path):
    import tifffile as tiff

    arr = tiff.imread(path)

    # Convert (C, H, W) to (H, W, C) when needed.
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))

    return arr


# ============================================================
# Helper Functions
# ============================================================

def read_image(img_path):
    """Read TIFF/PNG/JPG images with a display-compatible dtype."""
    ext = os.path.splitext(img_path)[1].lower()

    if ext in [".tif", ".tiff"]:
        img = tiff_imread(img_path)
    else:
        img = plt.imread(img_path)

    # Normalize floating-point images to 8-bit for display.
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img, 0, 1) * 255.0
        img = img.astype(np.uint8)

    elif img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)

    return img


def read_mask(mask_path, threshold=0.5):
    """Read and binarize a mask from a common image format."""
    ext = os.path.splitext(mask_path)[1].lower()

    if ext in [".tif", ".tiff"]:
        m = tiff_imread(mask_path)
    else:
        m = plt.imread(mask_path)

    # Convert RGB masks to a single channel.
    if m.ndim == 3:
        if m.shape[-1] == 1:
            m = m[..., 0]
        else:
            m = np.mean(m, axis=-1)

    m = m.astype(np.float32)

    # Normalize to 0-1 when needed.
    m_max = float(np.max(m)) if m.size else 0.0

    if m_max > 1.0:
        m = m / 255.0

    return (m > float(threshold)).astype(np.uint8)


def masks_to_coco_json(
    image_paths,
    mask_paths,
    save_path,
    category_id=1,
    threshold=0.5,
    min_area=1.0
):
    annotations, images = [], []
    ann_id = 1

    for img_id, (img_path, mask_path) in enumerate(
        zip(image_paths, mask_paths),
        start=1
    ):
        img = read_image(img_path)
        h, w = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": int(w),
            "height": int(h)
        })

        mask = read_mask(mask_path, threshold=threshold)
        labeled = measure.label(mask, connectivity=1)

        for region in measure.regionprops(labeled):
            if region.area < min_area:
                continue

            binary_mask = (labeled == region.label).astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(binary_mask))

            if isinstance(rle.get("counts"), bytes):
                rle["counts"] = rle["counts"].decode("utf-8")

            y1, x1, y2, x2 = region.bbox
            bbox = [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": float(region.area),
                "iscrowd": 0,
                "segmentation": rle
            })

            ann_id += 1

    coco_dict = {
        "info": {
            "description": "Dataset",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": category_id,
                "name": "object"
            }
        ]
    }

    os.makedirs(
        os.path.dirname(save_path) or ".",
        exist_ok=True
    )

    with open(save_path, "w") as f:
        json.dump(coco_dict, f)

    return save_path


def preds_to_coco_json(
    preds,
    image_paths,
    save_path,
    threshold=0.5,
    category_id=1,
    min_area=1.0
):
    """Resize predicted masks to original image size before encoding."""
    results = []
    ann_id = 1

    for img_id, (pred, img_path) in enumerate(
        zip(preds, image_paths),
        start=1
    ):
        pred = np.squeeze(pred).astype(np.float32)

        H, W = read_image(img_path).shape[:2]
        ph, pw = pred.shape[:2]

        # Resize predictions to the original image dimensions.
        if (ph, pw) != (H, W):
            pred = sk_resize(
                pred,
                (H, W),
                order=1,
                mode="reflect",
                anti_aliasing=False,
                preserve_range=True
            )
            pred = pred.astype(np.float32)

        # Apply the configured threshold.
        binary_mask = (pred > threshold).astype(np.uint8)

        # Fall back to Otsu thresholding if the mask is empty.
        if np.sum(binary_mask) == 0 and np.any(pred > 0):
            try:
                thr = float(threshold_otsu(pred))
                binary_mask = (pred > thr).astype(np.uint8)
            except Exception:
                pass

        labeled = measure.label(
            binary_mask,
            connectivity=1
        )

        for region in measure.regionprops(labeled):
            if region.area < min_area:
                continue

            mask_inst = (labeled == region.label).astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(mask_inst))

            if isinstance(rle.get("counts"), bytes):
                rle["counts"] = rle["counts"].decode("utf-8")

            y1, x1, y2, x2 = region.bbox
            coords = region.coords

            score = (
                float(np.mean(pred[coords[:, 0], coords[:, 1]]))
                if coords.size
                else float(np.max(pred))
            )

            results.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "score": score,
                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1)
                ],
                "segmentation": rle
            })

            ann_id += 1

    os.makedirs(
        os.path.dirname(save_path) or ".",
        exist_ok=True
    )

    with open(save_path, "w") as f:
        json.dump(results, f)

    return save_path


def run_coco_eval(gt_json, dt_json, logprint):
    if not os.path.exists(gt_json) or not os.path.exists(dt_json):
        logprint(
            f"[COCO] Missing file(s): "
            f"gt={os.path.exists(gt_json)} "
            f"dt={os.path.exists(dt_json)}"
        )
        return {}

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(dt_json)

    gt_anns = len(coco_gt.getAnnIds())
    dt_anns = len(coco_dt.getAnnIds())

    logprint(
        f"[COCO] GT anns: {gt_anns} | DT anns: {dt_anns}"
    )

    stats = {}

    for iou_type in ["bbox", "segm"]:
        coco_eval = COCOeval(
            coco_gt,
            coco_dt,
            iouType=iou_type
        )

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap, ap50, ap75 = [
            float(x) for x in coco_eval.stats[0:3]
        ]

        stats[f"AP_{iou_type}"] = ap
        stats[f"AP50_{iou_type}"] = ap50
        stats[f"AP75_{iou_type}"] = ap75

        logprint(
            f"[COCO-{iou_type}] AP: {ap:.4f} | "
            f"AP50: {ap50:.4f} | AP75: {ap75:.4f}"
        )

    return stats


def save_all_predictions(
    images,
    labels,
    preds,
    save_path,
    samples_per_image=2,
    threshold=0.5,
    generate_guiding_map=None
):
    os.makedirs(save_path, exist_ok=True)

    total_samples = len(images)
    num_groups = math.ceil(
        total_samples / samples_per_image
    )

    # Add an additional column when guidance is available.
    num_cols = 7 if generate_guiding_map is not None else 6

    for group_idx in range(num_groups):
        start = group_idx * samples_per_image
        end = min(
            start + samples_per_image,
            total_samples
        )
        n = end - start

        fig, axes = plt.subplots(
            n,
            num_cols,
            figsize=(4 * num_cols, 3 * max(1, n))
        )

        if n == 1:
            axes = axes[np.newaxis, :]

        for i in range(n):
            idx = start + i
            img, lbl, pred = (
                images[idx],
                labels[idx],
                preds[idx]
            )

            # Normalize original image for visualization.
            if img.dtype != np.uint8:
                imin, imax = float(img.min()), float(img.max())

                if imax > imin:
                    img_disp = (
                        (img - imin) /
                        (imax - imin) *
                        255.0
                    ).astype(np.uint8)
                else:
                    img_disp = (
                        img * 255.0
                    ).astype(np.uint8)
            else:
                img_disp = img

            lbl_disp = np.squeeze(lbl)
            pred_disp = np.squeeze(pred)

            col_offset = 0

            # Original image.
            axes[i, col_offset].imshow(img_disp)
            axes[i, col_offset].set_title("Original")
            axes[i, col_offset].axis("off")
            col_offset += 1

            # Guidance map.
            if generate_guiding_map is not None:
                guidance_map = generate_guiding_map(img_disp)

                axes[i, col_offset].imshow(
                    guidance_map[..., 0],
                    cmap="hot"
                )
                axes[i, col_offset].set_title("Guidance")
                axes[i, col_offset].axis("off")
                col_offset += 1

            # Ground-truth mask.
            axes[i, col_offset].imshow(
                lbl_disp,
                cmap="gray"
            )
            axes[i, col_offset].set_title("GT Mask")
            axes[i, col_offset].axis("off")
            col_offset += 1

            # Prediction.
            axes[i, col_offset].imshow(
                pred_disp,
                cmap="gray"
            )
            axes[i, col_offset].set_title("Prediction")
            axes[i, col_offset].axis("off")
            col_offset += 1

            # Prediction contour overlay.
            axes[i, col_offset].imshow(img_disp)
            axes[i, col_offset].contour(
                (pred_disp > threshold).astype(np.uint8),
                colors="r",
                linewidths=1
            )
            axes[i, col_offset].set_title("BBox Overlay")
            axes[i, col_offset].axis("off")
            col_offset += 1

            # Prediction segmentation overlay.
            axes[i, col_offset].imshow(
                img_disp,
                alpha=0.7
            )
            axes[i, col_offset].imshow(
                (pred_disp > threshold).astype(np.uint8),
                cmap="jet",
                alpha=0.4
            )
            axes[i, col_offset].set_title("Segm Overlay")
            axes[i, col_offset].axis("off")
            col_offset += 1

            # Edge map.
            edges = cv2.Canny(
                (pred_disp * 255).astype(np.uint8),
                50,
                150
            )

            axes[i, col_offset].imshow(
                edges,
                cmap="gray"
            )
            axes[i, col_offset].set_title("Edges")
            axes[i, col_offset].axis("off")
            col_offset += 1

        plt.tight_layout()

        save_file = os.path.join(
            save_path,
            f"outputs_{group_idx + 1:03d}.png"
        )

        plt.savefig(
            save_file,
            bbox_inches="tight"
        )
        plt.close()


# ============================================================
# Guided Generator
# ============================================================

class GuidedGenerator(tf.keras.utils.Sequence):

    def __init__(self, base_gen, guidance_fn):
        self.base_gen = base_gen
        self.guidance_fn = guidance_fn

    def __len__(self):
        return len(self.base_gen)

    def __getitem__(self, idx):
        imgs, lbls = self.base_gen[idx]
        guided_imgs = []

        for img in imgs:
            guidance = self.guidance_fn(img)
            img_with_guidance = np.concatenate(
                [img, guidance],
                axis=-1
            )
            guided_imgs.append(img_with_guidance)

        return np.array(guided_imgs), lbls


# ============================================================
# Main Test Loop
# ============================================================

print("\n=== Starting Test Evaluation with COCO ===")

log_file_path = os.path.join(
    summary_dir,
    "test_evaluation_logs_coco.txt"
)

logprint, test_log_file = get_logger(log_file_path)

best_fold = int(np.argmax(all_val_dice_scores)) + 1
best_dice = np.max(all_val_dice_scores)

best_model_path = os.path.join(
    model_dir,
    f"swinunet_model_kfold_{best_fold}.keras"
)

print(
    f"[BEST MODEL] Fold {best_fold} | "
    f"{best_model_path} "
    f"(Val Dice: {best_dice:.4f})"
)

config.enable_unsafe_deserialization()

model = load_model(
    best_model_path,
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
        "F1Score": F1Score,
        "ap_metric": ap_metric,
        "AFNR": AFNR
    }
)

test_subfolders = sorted(
    os.listdir(test_img_dir)
)

logprint(
    "\n=== Per-Folder Test Evaluation (with COCO) ==="
)

test_results, coco_results = [], []


# ============================================================
# Per-Folder Evaluation
# ============================================================

for subfolder in test_subfolders:
    img_dir = os.path.join(
        test_img_dir,
        subfolder
    )

    lbl_dir = os.path.join(
        test_lbl_dir,
        subfolder
    )

    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        continue

    img_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(
            ("png", "jpg", "jpeg", "tif", "tiff", "bmp")
        )
    ]

    lbl_files = [
        f for f in os.listdir(lbl_dir)
        if f.lower().endswith(
            (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")
        )
    ]

    img_dict = {
        os.path.splitext(f)[0]: os.path.join(img_dir, f)
        for f in img_files
    }

    lbl_dict = {
        os.path.splitext(f)[0]: os.path.join(lbl_dir, f)
        for f in lbl_files
    }

    common_keys = sorted(
        set(img_dict.keys()) & set(lbl_dict.keys())
    )

    if not common_keys:
        logprint(
            f"[SKIP] No matching image-mask pairs "
            f"in folder {subfolder}"
        )
        continue

    pairs = [
        (img_dict[k], lbl_dict[k])
        for k in common_keys
    ]

    base_generator = ImageLabelGenerator(
        pairs,
        batch_size=2,
        shuffle=False
    )

    generator = GuidedGenerator(
        base_generator,
        generate_guiding_map
    )

    all_images, all_labels, all_preds = [], [], []

    for idx in range(len(generator)):
        batch_images, batch_labels = generator[idx]
        batch_images_with_guidance = []

        batch_images_with_guidance = batch_images
        batch_images = np.array(batch_images_with_guidance)

        preds = model.predict(
            batch_images,
            verbose=0
        )

        all_images.extend(batch_images)
        all_labels.extend(batch_labels)
        all_preds.extend(preds)

    save_subfolder = os.path.join(
        output_dir,
        "test_output",
        f"{subfolder}_output"
    )

    os.makedirs(
        save_subfolder,
        exist_ok=True
    )

    gt_json = os.path.join(
        save_subfolder,
        f"{subfolder}_gt.json"
    )

    dt_json = os.path.join(
        save_subfolder,
        f"{subfolder}_dt.json"
    )

    # Export ground-truth and detection results to COCO format.
    masks_to_coco_json(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        gt_json,
        threshold=0.5
    )

    preds_to_coco_json(
        all_preds,
        [p[0] for p in pairs],
        dt_json,
        threshold=0.2
    )

    coco_stats = run_coco_eval(
        gt_json,
        dt_json,
        logprint
    )

    coco_results.append(coco_stats)

    results = model.evaluate(
        generator,
        verbose=0
    )

    test_results.append(results)

    # Unpack results according to model.metrics_names.
    try:
        loss, acc, dice, f1, ap, afnr = results
    except ValueError:
        logprint(
            f"[WARN] Unexpected number of metrics: "
            f"{len(results)} ({model.metrics_names})"
        )
        loss = acc = dice = f1 = ap = afnr = np.nan

    logprint(
        f"[{subfolder}] "
        f"Loss: {loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"Dice: {dice:.4f} | "
        f"F1: {f1:.4f} | "
        f"AP: {ap:.4f} | "
        f"AFNR: {afnr:.4f}"
    )

    save_all_predictions(
        np.array(all_images)[..., :3],
        np.array(all_labels),
        np.array(all_preds),
        save_subfolder,
        samples_per_image=2,
        threshold=0.2,
        generate_guiding_map=generate_guiding_map
    )


# ============================================================
# Overall Metrics
# ============================================================

metric_names = model.metrics_names

if test_results:
    logprint("\n=== Overall Test Metrics ===")

    for i, key in enumerate(metric_names):
        values = [tr[i] for tr in test_results]
        mean, std = np.mean(values), np.std(values)

        logprint(
            f"{key}: {mean:.4f} +/- {std:.4f}"
        )


# ============================================================
# Overall COCO Metrics
# ============================================================

if coco_results:
    ap_bbox_list, ap50_bbox_list, ap75_bbox_list = [], [], []
    ap_segm_list, ap50_segm_list, ap75_segm_list = [], [], []

    for c in coco_results:
        ap_bbox_list.append(
            float(c.get("AP_bbox", np.nan))
        )
        ap50_bbox_list.append(
            float(c.get("AP50_bbox", np.nan))
        )
        ap75_bbox_list.append(
            float(c.get("AP75_bbox", np.nan))
        )
        ap_segm_list.append(
            float(c.get("AP_segm", np.nan))
        )
        ap50_segm_list.append(
            float(c.get("AP50_segm", np.nan))
        )
        ap75_segm_list.append(
            float(c.get("AP75_segm", np.nan))
        )

    logprint("\n=== Overall COCO Test Metrics ===")

    logprint(
        f"COCO_bbox_AP: "
        f"{np.nanmean(ap_bbox_list):.4f} +/- "
        f"{np.nanstd(ap_bbox_list):.4f}"
    )

    logprint(
        f"COCO_bbox_AP50: "
        f"{np.nanmean(ap50_bbox_list):.4f} +/- "
        f"{np.nanstd(ap50_bbox_list):.4f}"
    )

    logprint(
        f"COCO_bbox_AP75: "
        f"{np.nanmean(ap75_bbox_list):.4f} +/- "
        f"{np.nanstd(ap75_bbox_list):.4f}"
    )

    logprint(
        f"COCO_segm_AP: "
        f"{np.nanmean(ap_segm_list):.4f} +/- "
        f"{np.nanstd(ap_segm_list):.4f}"
    )

    logprint(
        f"COCO_segm_AP50: "
        f"{np.nanmean(ap50_segm_list):.4f} +/- "
        f"{np.nanstd(ap50_segm_list):.4f}"
    )

    logprint(
        f"COCO_segm_AP75: "
        f"{np.nanmean(ap75_segm_list):.4f} +/- "
        f"{np.nanstd(ap75_segm_list):.4f}"
    )


# ============================================================
# Finalize Test Evaluation
# ============================================================

logprint(
    f"\n[FINISHED] Summary written to: {log_file_path}"
)

test_log_file.close()
