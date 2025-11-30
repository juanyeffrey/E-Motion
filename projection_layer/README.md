# Projection Layer Training Guide

This folder contains the code to train the **Perception Projection Layer**. This layer maps the 1441-dimensional perception vector (facial landmarks + emotions) into the 768-dimensional CLIP text embedding space used by Stable Diffusion.

## 1. Data Requirements

To train this model, you need a dataset of paired **Perception Vectors** and **CLIP Text Embeddings**.

### Format
The training script expects a single `.npz` file containing two numpy arrays:

1.  **`perception_embeddings`**: Shape `(N, 1441)`
    *   Input features extracted from your video/image dataset using the `PerceptionPipeline`.
    *   Contains flattened facial landmarks (478 * 3 = 1434) + emotion probabilities (7).
2.  **`clip_embeddings`**: Shape `(N, 768)`
    *   Target embeddings generated from text descriptions of the emotions/expressions in the videos.
    *   Generated using the `openai/clip-vit-large-patch14` text encoder.

## 2. How to Create the Dataset

I have provided a helper script `create_dataset_template.py` in this folder. You can use it as a starting point.

### Steps:
1.  **Collect Data**: Gather a set of images or video frames representing different emotions (e.g., RAVDESS, FER2013, or your own recordings).
2.  **Label Data**: Assign a descriptive text prompt to each image (e.g., "a person looking surprised", "angry facial expression").
3.  **Run Extraction**:
    *   Pass the image through `PerceptionPipeline` to get the **1441D vector**.
    *   Pass the text prompt through the **CLIP Text Encoder** to get the **768D vector**.
4.  **Save**: Aggregate all vectors into the `.npz` file.

## 3. Training the Model

Once you have your `training_data.npz`, run the training script:

```bash
# Run from the root directory of the project
python -m projection_layer.train_projection --data path/to/training_data.npz --epochs 100 --batch_size 32
```

### Arguments:
*   `--data`: Path to your `.npz` dataset file.
*   `--out_dir`: Where to save the model (default: `projection_layer/checkpoints`).
*   `--epochs`: Number of training epochs (default: 100).
*   `--lr`: Learning rate (default: 1e-4).

## 4. Integration

The training script will save the best model to `projection_layer/checkpoints/best_model.pth`.

**To use it in the app:**
1.  Ensure `best_model.pth` is in `projection_layer/checkpoints/`.
2.  The app will automatically detect and load it on startup.
