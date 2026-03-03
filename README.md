# GeoSpatial Segmentation — U-Net on LandCover.ai

A PyTorch implementation of U-Net for multi-class semantic segmentation of aerial/satellite imagery, trained on the [LandCover.ai v1](https://landcover.ai/) dataset. The model segments geospatial imagery into 5 land-cover classes.

---

## Classes

| ID | Class | Colour |
|----|------------|------------------|
| 0 | Background | ⬛ |
| 1 | Building | 🟥 |
| 2 | Woodlands | 🟩 |
| 3 | Water | 🟦 |
| 4 | Road | 🟨 |

---

## Model Architecture

A custom U-Net with:
- **Encoder**: 4 downsampling blocks with feature sizes `[32, 64, 128, 256]`
- **Bottleneck**: Double convolution at `512` channels
- **Decoder**: 4 upsampling blocks with transposed convolutions + skip connections
- **Output**: 1×1 convolution → 5-class segmentation map
- **Input size**: 512×512 RGB images

Each block uses double 3×3 convolutions with Batch Normalisation and ReLU activations.

---

## 📦 Dataset & Preprocessing

- **Dataset**: [LandCover.ai v1](https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip)
- Large `.tif` images are tiled into **512×512 chips**
- **Split**: 70% train / 15% validation / 15% test

### Augmentations (training only)
- Random brightness & contrast
- Horizontal & vertical flips
- Median blur

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch size | 8 |
| Optimiser | AdamW |
| Scheduler | OneCycleLR (three-phase) |
| Max LR | 6.43e-3 |
| Loss | Hybrid (Dice + weighted CrossEntropy) |

The model is saved whenever a new best validation F1 is achieved.

---

## 📊 Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| Loss | 0.6668 |
| Accuracy | 93.01% |
| F1-Score (macro) | 0.7820 |
| Cohen's Kappa | 0.8772 |

### Training Curves

![Loss and F1 curves over 30 epochs](loss_curve_30epoch_0maxlr_0_7600372best_f1.png)

Training and validation loss converge steadily after epoch 15. The sharp dip in validation F1 around epoch 10 corresponds to the OneCycleLR peak phase, after which the model recovers strongly and plateaus near **0.75+ validation F1**.

---

## 🚀 Getting Started

### Requirements

- torch
- torchvision
- kornia
- torchmetrics
- torch-lr-finder
- albumentations
- opencv-python
- pandas
- matplotlib
- numpy

### Data Preparation

```bash
mkdir -p landcover_data/images landcover_data/masks landcover_data/chips
wget -q https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip
unzip -q landcover.ai.v1.zip -d landcover_data
```
### Inference

```python
model = UNET(in_channels=3, out_channels=5, features=[32, 64, 128, 256]).to(device)
model.load_state_dict(torch.load("GS_Seg_UNET_model.pt", map_location=device))
model.eval()

with torch.no_grad():
    pred = model(image_tensor.unsqueeze(0))
    mask = torch.argmax(pred, dim=1)
```
---

## Reference:
This project uses the [LandCover.ai dataset](https://landcover.ai/)
