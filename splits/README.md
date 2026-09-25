# Dataset Split Documentation

## Training Dataset: Dataset_HI

To evaluate the generalization capability of the detectors and reduce generator-specific bias, we construct a unified dataset denoted as **Dataset_HI** by combining two publicly available datasets:

### Source 1: Human-Faces-Dataset
- A facial image dataset containing both authentic and synthetic images
- Contributes 10,000 images (5,000 authentic + 5,000 synthetic)

### Source 2: ImageNet-1K-based DeepFake Detection Dataset
- Constructed from a 10-class subset of ImageNet-1K
- Authentic images taken from selected ImageNet-1K classes
- Synthetic images generated using a pretrained class-conditional latent diffusion model (LDM)
- Contributes 10,000 images (5,000 authentic + 5,000 synthetic)

### Dataset_HI Statistics
- **Total**: 20,000 images
- **Authentic**: 10,000 images
- **Synthetic**: 10,000 images
- **Balanced within each source**: Neither source alone predicts the real/fake label

### Split Ratio (7:2:1)
| Split | Number of Images | Real | Fake |
|-------|-----------------|------|------|
| Training | 14,000 | 7,000 | 7,000 |
| Validation | 4,000 | 2,000 | 2,000 |
| Test | 2,000 | 1,000 | 1,000 |

- Partition performed at the image level
- No image is shared across splits
- Balanced class distribution preserved in each split

### Directory Structure
```
Dataset_HI/
├── train/
│   ├── 0_real/    # 7,000 authentic images
│   └── 1_fake/    # 7,000 synthetic images
├── val/
│   ├── 0_real/    # 2,000 authentic images
│   └── 1_fake/    # 2,000 synthetic images
└── test/
    ├── 0_real/    # 1,000 authentic images
    └── 1_fake/    # 1,000 synthetic images
```

## Cross-generator Testing Datasets

Seven testing datasets not involved in training, covering five major generative paradigms:

| Dataset | Generative Paradigm | Generator |
|---------|-------------------|-----------|
| DALL-E | Autoregressive models | DALL-E |
| Glide50 | Text-guided diffusion | GLIDE (50% guidance) |
| Glide100+10 | Text-guided diffusion | GLIDE (100% + 10% steps) |
| Glide100+27 | Text-guided diffusion | GLIDE (100% + 27% steps) |
| Guided Diffusion | Classifier-guided diffusion | Guided Diffusion |
| LDM200+CFG | Latent diffusion models | LDM (200 steps + CFG) |
| WildFake-SD | New-generation open-source diffusion | SD v1.5, SD-ControlNet, SD-LoRA, SD-LyCORIS |

### Testing Protocol
- All detectors are trained solely on the training split of Dataset_HI
- Directly evaluated on these testing subsets without any additional fine-tuning
- Ensures fair assessment of each method's ability to generalize across different generative mechanisms

## Evaluation Metrics
- Average Precision (AP)
- Accuracy (ACC)
- Area Under the ROC Curve (AUROC)
- F1 Score
