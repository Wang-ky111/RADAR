# Baseline Reproduction Settings

All baseline methods are reimplemented from their publicly available open-source code. Following the training protocols, preprocessing, input resolution, optimizer, and hyperparameter settings reported in the original papers or their official repositories, each baseline is retrained from scratch on our **Dataset_HI** rather than directly evaluated with the official released weights. All baselines are evaluated under the same training-validation-test split and the same data preprocessing pipeline as RADAR.

## Unified Experimental Settings

- **Training Dataset**: Dataset_HI (14,000 training images, 7,000 real + 7,000 fake)
- **Validation Dataset**: Dataset_HI validation split (4,000 images)
- **Testing Datasets**: Dataset_HI test split + 7 cross-generator datasets
- **Input Resolution**: 256×256 (unless specified otherwise by the original paper)
- **Evaluation Metrics**: AP, ACC, AUROC, F1

---

## 1. CNN-Spot (Wang et al., CVPR 2020)

**Architecture**: ResNet-50 based classifier with carefully designed preprocessing, post-processing, and blur/JPEG augmentations.

**Key Settings** (following original paper):
- Backbone: ResNet-50
- Input size: 256×256
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 32
- Data augmentation: Gaussian blur (probability 0.1), JPEG compression (probability 0.1)
- Training epochs: 50

--

## 2. Freq-Mask (Doloriel et al., ICASSP 2024)

**Architecture**: ResNet-50 with supervised frequency masking during training to learn robust representations for unseen generators.

**Key Settings** (following original paper):
- Backbone: ResNet-50
- Input size: 256×256
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 32
- Frequency masking ratio: 15%
- Training epochs: 50

---

## 3. ETD (Ba et al., AAAI 2024)

**Architecture**: Local forgery cue extraction with disentangled representation learning and global feature aggregation.

**Key Settings** (following original paper):
- Backbone: ResNet-50
- Input size: 256×256
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 32
- Local information blocks: K=4
- Training epochs: 50

---

## 4. MIFAE (Wang et al., ICASSP 2025)

**Architecture**: Two-stage masked image-frequency autoencoder that jointly models spatial consistency via masked image modeling and frequency relationships from low- to high-frequency components, further fine-tuned with linear probing.

**Key Settings** (following original paper):
- Backbone: MAE ViT-Base (patch size 16)
- Input size: 224×224
- Optimizer: AdamW
- Learning rate: 1.5e-4 (pretraining), 1e-3 (fine-tuning)
- Batch size: 64 (pretraining), 32 (fine-tuning)
- FFT mask radius: 16
- Training epochs: 200 (pretraining), 50 (fine-tuning)

---

## 5. UFD (Ojha et al., CVPR 2023)

**Architecture**: Universal fake image detector built on a frozen CLIP-ViT feature space, leveraging cosine-distance nearest-neighbor classification and linear probing.

**Key Settings** (following original paper):
- Backbone: CLIP ViT-B/16 (frozen)
- Input size: 224×224
- Classifier: Linear probing + nearest-neighbor
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 32
- Training epochs: 50

---

## 6. CO-SPY (Cheng et al., CVPR 2025)

**Architecture**: Synthetic image detector that adaptively fuses frozen CLIP semantic features with VAE reconstruction artifacts via learnable regulators, using feature-space interpolation and random dropout to reduce reliance on generator-specific patterns.

**Key Settings** (following original paper):
- Backbone: CLIP ViT-B/16 (frozen) + VAE
- Input size: 256×256
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 32
- Training epochs: 50

---

## 7. SPAI (Karageorgiou et al., CVPR 2025)

**Architecture**: Any-resolution detector that learns the spectral distribution of real images through self-supervised masked spectral modeling and identifies AI-generated images by comparing spectral reconstruction similarities across original and low-/high-pass filtered representations.

**Key Settings** (following original paper):
- Backbone: Spectral learning network
- Input size: Any resolution (tested at 256×256)
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 32
- Training epochs: 50

---

