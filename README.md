# TriNetra: A Structurally Informed Transformer with Augmentation-Driven Curriculum Learning for Cell Segmentation

This repository contains a modular implementation of **Tri(Three)** **Netra(Vision)**, a transfomer framework for cell segmentation in microscopy images.

TriNetra integrates three complementary components:

- **Augmentation-based Curriculum Learning Approach** to progressively increase augmentation strength during training.
- **Structural prior guidance** to provide additional spatial information.
- **Hierarchical boundary representations** to enhance boundary-sensitive feature learning and improve cell delineation across different cellular growth stages.

The code is organized into separate components for clarity, reusability, and reproducibility, allowing you to either:

- Run the modules independently and orchestrate training/evaluation from your own `main.py`, or
- Combine the components into a single pipeline for experimentation.

---

## Project Structure

The `src/` directory contains the main components:

    imports.py
        Centralized imports and global configurations, including dataset paths, output directories, and runtime settings.

    dataloader.py
        Dataset utilities for pairing images and labels, loading microscopy images, preprocessing masks, and integrating structural prior guidance.

     augmentation.py
        Augmentation-based Curriculum Learning that adjusts the augmentation ratio progressively across training epochs.

    model.py
        TriNetra architecture based on SwinUNet, including transformer blocks, patch operations, skip connections, structural piors integration, and hierarchical boundary representations.

    metrics.py
        Training losses and evaluation metrics, including BCE + Dice, Dice coefficient, F1 score, Average Precision, and AFNR.

    structural_prior_and_boundary_generation.py
        Generates structural prior maps from phase-contrast microscopy images.
        
    callbacks_checks.py
        Custom callbacks for training monitoring, checkpointing, prediction visualization, sanity checks, curriculum learning, and COCO-based evaluation.

    test.py
        Test-time evaluation, prediction generation, visualization, and COCO-style object detection and segmentation metrics.

    train.py
        Training pipeline using k-fold cross-validation, model training, validation, metric tracking, and result(aggregated) generation.

  Other files in the root directory:

    .gitignore – To ignore checkpoints, outputs, and logs.
    README.md – This documentation file.
    download_model.py – Download the pretrained TriNetra model from Hugging Face.


---

## TriNetra Architecture

TriNetra is designed around three complementary components:

### 1. Transformer Backbone with Augmentation-based Curriculum Learning

#### Augmentation-based Curriculum Learning(ACL) Approach:

<img width="13089" height="7157" alt="CL" src="https://github.com/user-attachments/assets/1c11ce30-c9e3-4dfe-b370-d99c581ad991" />

#### The core segmentation network uses a hierarchical Swin Transformer-based encoder-decoder architecture (Base Arch: [GPiSeT](https://github.com/UrjitMehta/GPiSeT.git)):

<img width="21741" height="11243" alt="CL_Model" src="https://github.com/user-attachments/assets/42599b9f-fe64-4c6c-a6e7-d9c4641bb0f0" />

The encoder progressively captures contextual information at multiple spatial resolutions, while the decoder reconstructs high-resolution segmentation maps through patch expansion and skip connections. Augmentation-based Curriculum Learning (ACL) progressively adjusts the augmentation ratio across training epochs to increase training difficulty as learning progresses.


---

### 2. Structural Prior Guidance

TriNetra incorporates a **structural prior** as an additional input-level representation.

The structural prior provides spatial information derived from the phase-contrast microscopy image and is concatenated with the original image before being processed by the segmentation network.

This provides the model with an additional representation that can complement the appearance-based features learned by the transformer backbone.

The structural prior is generated using the implementation provided in:

    src/structural_prior_and_boundary_generation.py

The resulting guidance representation is integrated into the data-loading and model-input pipeline.

---

### 3. Hierarchical Boundary Representations

TriNetra incorporates **hierarchical boundary representations** within the segmentation architecture.

Rather than relying only on the final segmentation output, boundary-related information is integrated across multiple feature levels of the network.

This allows the model to preserve and refine boundary information throughout the encoder-decoder hierarchy, which is particularly relevant for densely packed and adjacent cells where accurate delineation is required.

---

### Qualitative Results By TriNetra

<img width="12576" height="6754" alt="LinkedIN POST" src="https://github.com/user-attachments/assets/17a58d9e-4bf9-40a8-935b-abbd1fdc0fdd" />

---

## Configuration

Dataset paths, output directories, model directories, logging paths, and runtime configuration are defined in:

    src/imports.py

The primary dataset paths that may need to be configured are:

    train_img_dir
    train_lbl_dir
    test_img_dir
    test_lbl_dir

Additional configuration includes:

- Model output directories
- Training logs
- TensorBoard logs
- Evaluation outputs
- Cross-validation outputs
- Mixed-precision configuration
- Input prior configuration

Structural prior guidance can be enabled using:

    use_input_guidance = True

---

## Training

TriNetra uses **k-fold cross-validation** for model training and evaluation.

The training pipeline performs:

    Prepare image-label pairs

    Configure k-fold cross-validation

    Create training and validation generators

    Generate structural prior guidance

    Initialize TriNetra

    Configure optimizer and learning-rate schedule

    Compile the model

    Configure training callbacks

    Train the model

    Evaluate validation performance

    Store training histories

    Generate COCO evaluation metrics

    Generate training curves

    Save fold-specific models and outputs
    
---

## Usage

1. **Install all/core dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    OR
   ```bash
   pip install -r core_req.txt
   ```

3. **Pretrained Model:**

    - A pretrained version of the proposed **TriNetra** model will be made available on Hugging Face.

    - The pretrained model can be used for testing or inference on compatible microscopy datasets.

    - **Download directly from Hugging Face:**

        The pretrained `TriNetra.keras` model can be downloaded directly from the [Hugging Face Model Repository](https://huggingface.co/urjit006/TriNetra) and used according to your requirements.

    - **Fine-tune from the TriNetra architecture:**

        The **TriNetra** architecture can be initialized from `src/model.py` and fine-tuned or trained on a custom dataset. Configure the required dataset and output paths in `src/imports.py` before training.

4. **Dataset:**

    - Prepare the dataset according to the directory structure expected by the data loader:

        ```text
        dataset/
        └── combined_dataset/
            ├── train_images/
            ├── train_all_TIFF_labels/
            ├── test_images/
            └── test_all_TIFF_labels/
        ```
    - We combined a total of 5 publicly available datasets for training:
        1. [LIVECell Dataset](https://sartorius-research.github.io/LIVECell/)(Evaluation Benchmark)
        2. [Data Science Bowl 2018](https://bbbc.broadinstitute.org/BBBC038)
        3. [Cellpose](https://www.cellpose.org/)
        4. [NeurIPS 2022 Cell Segmentation Challenge dataset](https://neurips22-cellseg.grand-challenge.org/dataset/)
       

5. **Procedure:**

    - Import modules
    - Prepare datasets
    - Generate structural prior guidance
    - Define the TriNetra model
    - Add metrics
    - Configure Augmentation-based Curriculum Learning
    - Set up callbacks, logging, and checks
    - Train the model
    - Test/evaluate the results

## Notes

Each module is standalone; you can import functions/classes as needed:

```python
from dataloader import get_image_label_pairs, ImageLabelGenerator
from model import swin_unet
from metrics import bce_dice_loss, dice_coef, F1Score, ap_metric, AFNR
```
---
## Citation

If you use this repository or find the work helpful, please cite the paper:

```bibtex
@inproceedings{um2026trinetra,
  title={TriNetra: A Structurally Informed Transformer with Augmentation-Driven Curriculum Learning for Cell Segmentation},
  author={Mehta, Urjit and Bhalodiya, Jayendra M.},
  booktitle={14th International Conference on Big Data Analytics and Artificial Intelligence (BDA2026)},
  year={2026}
  url={https://github.com/UrjitMehta/TriNetra}
}
```

