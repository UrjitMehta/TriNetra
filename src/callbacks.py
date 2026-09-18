# ===============================
# Callback: Output Sanity Check
# ===============================

class OutputSanityCheckCallback(tf.keras.callbacks.Callback):

    def __init__(self, generator, num_batches_to_check=1):
        self.generator = generator
        self.num_batches_to_check = num_batches_to_check

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"\n[Sanity Check] Checking model outputs after epoch {epoch + 1}"
        )

        for batch_idx in range(self.num_batches_to_check):
            images, _ = self.generator[batch_idx]
            preds = self.model.predict(images)

            # Check for NaN or Inf.
            if np.isnan(preds).any():
                print(
                    f"[ERROR] Model output contains NaN in batch {batch_idx}!"
                )

            if np.isinf(preds).any():
                print(
                    f"[ERROR] Model output contains Inf in batch {batch_idx}!"
                )

            # Check prediction range.
            min_pred = preds.min()
            max_pred = preds.max()

            print(
                f"[INFO] Batch {batch_idx}: "
                f"pred min={min_pred:.5f}, max={max_pred:.5f}"
            )

            # Check if model is predicting all zeros or all ones.
            if np.all(preds == 0):
                print(
                    f"[WARNING] Model predicts all ZEROS in batch {batch_idx}"
                )

            if np.all(preds == 1):
                print(
                    f"[WARNING] Model predicts all ONES in batch {batch_idx}"
                )


# ===============================
# Callback: K-Fold Checkpoint
# ===============================

class EpochCheckpoint(Callback):

    def __init__(self, save_path_base, fold_number, interval=10):
        super().__init__()

        self.save_path_base = save_path_base
        self.fold_number = fold_number
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            model_dir = os.path.join(
                self.save_path_base,
                f"fold_{self.fold_number}"
            )

            os.makedirs(model_dir, exist_ok=True)

            model_path = os.path.join(
                model_dir,
                f"swin_model_epoch{epoch + 1}.keras"
            )

            self.model.save(model_path)

            print(
                f"\n[INFO] Saved model at epoch {epoch + 1} "
                f"to {model_path}"
            )


# =======================================
# Callback: Save Predictions After Each Epoch
# =======================================

class SavePredictionsCallback(tf.keras.callbacks.Callback):
    """
    Save random validation predictions each epoch for visual monitoring.

    Args:
        generator: Data generator returning (images, labels).
        save_path: Output directory.
        num_sets: Number of samples to visualize.
        threshold: Mask binarization threshold.
        generate_guiding_map: Optional function returning guidance maps.
        edge_extractor: Optional model to extract edges.
    """

    def __init__(
        self,
        generator,
        save_path,
        num_sets=2,
        threshold=0.5,
        generate_guiding_map=None,
        edge_extractor=None
    ):
        super().__init__()

        self.generator = generator
        self.save_path = save_path
        self.num_sets = num_sets
        self.num_batches = len(generator)
        self.threshold = threshold
        self.generate_guiding_map = generate_guiding_map
        self.edge_extractor = edge_extractor

    def on_epoch_end(self, epoch, logs=None):
        os.makedirs(self.save_path, exist_ok=True)

        batch_index = random.randint(
            0,
            self.num_batches - 1
        )

        images, labels = self.generator[batch_index]

        predictions = self.model.predict(
            images,
            verbose=0
        )

        edge_maps = None

        if self.edge_extractor is not None:
            edge_maps = self.edge_extractor.predict(
                images,
                verbose=0
            )

        num_samples_to_plot = min(
            self.num_sets,
            len(images)
        )

        # Dynamic number of columns.
        num_cols = 5

        if self.generate_guiding_map is not None:
            num_cols += 1

        if edge_maps is not None:
            num_cols += 1

        fig, axes = plt.subplots(
            num_samples_to_plot,
            num_cols,
            figsize=(
                4 * num_cols,
                4 * num_samples_to_plot
            )
        )

        axes = np.atleast_2d(axes)

        for i in range(num_samples_to_plot):
            absolute_idx = (
                batch_index * self.generator.batch_size + i
            )

            if absolute_idx >= len(self.generator.indexes):
                continue

            img_idx = self.generator.indexes[absolute_idx]
            img_path = self.generator.image_filenames[img_idx]
            col = 0

            # Load original image.
            try:
                ext = os.path.splitext(img_path)[-1].lower()

                if ext in [".tif", ".tiff"]:
                    img = tiff.imread(img_path)

                    if img.ndim == 2:
                        img = np.stack([img] * 3, axis=-1)

                    elif img.shape[0] in [1, 3]:
                        img = np.transpose(
                            img,
                            (1, 2, 0)
                        )

                    img = img[..., :3]

                else:
                    img = load_img(
                        img_path,
                        target_size=self.generator.target_size
                    )

                    img = img_to_array(img)

                img = img.astype(np.uint8)
                img_norm = img / 255.0

                axes[i, col].imshow(img_norm)
                axes[i, col].set_title("Original")

            except (
                FileNotFoundError,
                UnidentifiedImageError,
                OSError
            ):
                axes[i, col].text(
                    0.5,
                    0.5,
                    "Load Error",
                    ha="center"
                )

            axes[i, col].axis("off")
            col += 1

            # Guidance map.
            if self.generate_guiding_map is not None:
                guide = self.generate_guiding_map(img)

                axes[i, col].imshow(
                    guide[..., 0],
                    cmap="gray"
                )

                axes[i, col].set_title("Guidance")
                axes[i, col].axis("off")

                col += 1

            # Ground truth.
            axes[i, col].imshow(
                labels[i].squeeze(),
                cmap="gray"
            )

            axes[i, col].set_title("Label")
            axes[i, col].axis("off")

            col += 1

            # Prediction.
            pred_mask = np.squeeze(predictions[i])

            axes[i, col].imshow(
                pred_mask,
                cmap="gray"
            )

            axes[i, col].set_title("Prediction")
            axes[i, col].axis("off")

            col += 1

            # Bounding boxes.
            pred_bin = (
                pred_mask > self.threshold
            ).astype(np.uint8)

            labeled = label(pred_bin)

            axes[i, col].imshow(img_norm)

            for region in regionprops(labeled):
                y1, x1, y2, x2 = region.bbox

                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    edgecolor="red",
                    facecolor="none",
                    linewidth=2
                )

                axes[i, col].add_patch(rect)

            axes[i, col].set_title("Bounding Box")
            axes[i, col].axis("off")

            col += 1

            # Overlay.
            axes[i, col].imshow(img_norm)

            axes[i, col].imshow(
                pred_bin,
                cmap="jet",
                alpha=0.4
            )

            axes[i, col].set_title("Overlay")
            axes[i, col].axis("off")

            col += 1

            # Refined Sobel edge.
            edge = refined_sobel_edges(img)

            axes[i, col].imshow(
                edge,
                cmap="gray"
            )

            axes[i, col].set_title("Refined Sobel")
            axes[i, col].axis("off")

        plt.tight_layout()

        save_file = os.path.join(
            self.save_path,
            f"epoch_{epoch + 1:03d}.png"
        )

        plt.savefig(
            save_file,
            bbox_inches="tight"
        )

        plt.close()


# =======================================
# Callback: COCO Evaluation Per Epoch
# =======================================

class COCOEvalCallback(tf.keras.callbacks.Callback):

    def __init__(
        self,
        val_generator,
        save_dir,
        threshold=0.1,
        eval_interval=5
    ):
        """
        Args:
            val_generator: Keras Sequence validation generator
                returning (images, masks).
            save_dir: Directory for temporary JSON files.
            threshold: Threshold for binarizing predictions.
            eval_interval: Run COCO evaluation every N epochs.
        """
        super().__init__()

        self.val_generator = val_generator
        self.save_dir = save_dir
        self.threshold = threshold
        self.eval_interval = eval_interval

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        self.last_logs = {}

    def _masks_to_instances(self, binary_mask):
        return label(
            binary_mask,
            connectivity=2
        )

    def _instances_to_coco_anns(
        self,
        instance_mask,
        image_id,
        start_ann_id=0,
        category_id=1,
        is_pred=False,
        scores=None
    ):
        anns = []
        ann_id = start_ann_id
        props = regionprops(instance_mask)

        for i, p in enumerate(props):
            if p.area < 2:
                continue

            mask_bin = (
                instance_mask == p.label
            ).astype(np.uint8)

            rle = maskUtils.encode(
                np.asfortranarray(mask_bin)
            )

            rle["counts"] = rle["counts"].decode("utf-8")

            bbox = [
                p.bbox[1],
                p.bbox[0],
                p.bbox[3] - p.bbox[1],
                p.bbox[2] - p.bbox[0]
            ]

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": int(p.area),
                "iscrowd": 0,
                "segmentation": rle
            }

            if is_pred:
                ann["score"] = float(
                    scores[i] if scores is not None else 1.0
                )

            anns.append(ann)
            ann_id += 1

        return anns, ann_id

    def on_epoch_end(self, epoch, logs=None):

        if logs is None:
            logs = {}

        if (epoch + 1) % self.eval_interval != 0:
            return

        print(
            f"\n[COCOEval] Running COCO evaluation "
            f"at epoch {epoch + 1}..."
        )

        all_images = []
        all_gt = []
        all_preds = []

        for b in range(len(self.val_generator)):
            imgs, lbls = self.val_generator[b]

            preds = self.model.predict(
                imgs,
                verbose=0
            )

            all_images.append(imgs)
            all_gt.append(lbls)
            all_preds.append(preds)

        all_images = np.concatenate(
            all_images,
            axis=0
        )

        all_gt = np.concatenate(
            all_gt,
            axis=0
        )

        all_preds = np.concatenate(
            all_preds,
            axis=0
        )

        # Build COCO-style ground truth and predictions.
        gt_json = {
            "info": {
                "description": "Validation dataset",
                "version": "1.0"
            },
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "cell"
                }
            ]
        }

        pred_json = []

        ann_id = 0
        image_id = 0

        for i in range(all_gt.shape[0]):
            gt_mask = (
                all_gt[i, ..., 0] > 0.5
            ).astype(np.uint8)

            pred_mask = (
                all_preds[i, ..., 0] >= self.threshold
            ).astype(np.uint8)

            gt_instances = self._masks_to_instances(
                gt_mask
            )

            pred_instances = self._masks_to_instances(
                pred_mask
            )

            h, w = gt_mask.shape

            gt_json["images"].append(
                {
                    "id": image_id,
                    "width": w,
                    "height": h,
                    "file_name": f"val_img_{image_id}.png"
                }
            )

            gt_anns, ann_id = self._instances_to_coco_anns(
                gt_instances,
                image_id,
                ann_id,
                is_pred=False
            )

            gt_json["annotations"].extend(
                gt_anns
            )

            pred_anns, _ = self._instances_to_coco_anns(
                pred_instances,
                image_id,
                0,
                is_pred=True
            )

            pred_json.extend(pred_anns)

            image_id += 1

        # Save temporary JSON files.
        gt_path = os.path.join(
            self.save_dir,
            f"gt_epoch{epoch + 1}.json"
        )

        with open(gt_path, "w") as f:
            json.dump(gt_json, f)

        pred_path = os.path.join(
            self.save_dir,
            f"pred_epoch{epoch + 1}.json"
        )

        with open(pred_path, "w") as f:
            json.dump(pred_json, f)

        if not gt_json["annotations"] or not pred_json:
            print(
                "[COCOEval] Skipping, no annotations found."
            )
            return

        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pred_path)

        # Evaluate bounding boxes and segmentation.
        for eval_type in ["bbox", "segm"]:
            coco_eval = COCOeval(
                coco_gt,
                coco_dt,
                iouType=eval_type
            )

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Logs: AP@[.5:.95], AP50, AP75.
            logs[f"AP_{eval_type}"] = coco_eval.stats[0]
            logs[f"AP50_{eval_type}"] = coco_eval.stats[1]
            logs[f"AP75_{eval_type}"] = coco_eval.stats[2]

        print(
            f"[COCOEval] Epoch {epoch + 1} - "
            f"bbox AP: {logs['AP_bbox']:.4f}, "
            f"segm AP: {logs['AP_segm']:.4f}"
        )

        self.last_logs = {
            "coco_bbox_ap": logs.get("AP_bbox", 0.0),
            "coco_segm_ap": logs.get("AP_segm", 0.0),
            "coco_bbox_ap50": logs.get("AP50_bbox", 0.0),
            "coco_bbox_ap75": logs.get("AP75_bbox", 0.0),
            "coco_segm_ap50": logs.get("AP50_segm", 0.0),
            "coco_segm_ap75": logs.get("AP75_segm", 0.0),
        }


# =======================================
# Callback: Curriculum Learning
# =======================================

class CurriculumLearningCallback(tf.keras.callbacks.Callback):
    """
    Gradually increases augmentation strength during training epochs.
    """

    def __init__(
        self,
        total_epochs,
        mode="cosine",
        max_strength=0.7
    ):
        super().__init__()

        self.total_epochs = total_epochs
        self.mode = mode
        self.max_strength = max_strength

    def on_epoch_begin(self, epoch, logs=None):
        global current_augmentation_strength

        progress = epoch / float(
            self.total_epochs - 1
        )

        if self.mode == "cosine":
            # Smooth cosine increase.
            base = 0.5 * (
                1 - np.cos(np.pi * progress)
            )
        else:
            # Linear ramp-up.
            base = progress

        current_augmentation_strength = (
            self.max_strength * base
        )

        print(
            f"[CurriculumLearning] Epoch {epoch + 1}: "
            f"Aug Strength = {current_augmentation_strength:.2f}"
        )
