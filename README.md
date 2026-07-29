# TinyVGG-vs-ResNet18
# EuroSAT CNN Model Comparison

A comparison of four CNN training setups on the [EuroSAT](https://github.com/phelber/EuroSAT) land-cover classification dataset (10 classes, RGB satellite tiles):

- TinyVGG — trained from scratch, no augmentation
- TinyVGG — trained from scratch, with augmentation
- ResNet18 — ImageNet-pretrained, fine-tuned, no augmentation
- ResNet18 — ImageNet-pretrained, fine-tuned, with augmentation

Each model uses the same 70/15/15 train/val/test split and is evaluated on accuracy, loss, a confusion matrix, and a classification report.

**Colab notebook:** https://colab.research.google.com/drive/1DxuTBYpita_DydJ3erONeNwGlgi2H_l9?usp=sharing

## Clone

```bash
git clone <https://github.com/Xebel097/TinyVGG-vs-ResNet18>
cd <TinyVGG-vs-ResNet18>
```

## Setup

1. Install dependencies:
   ```bash
   pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn kaggle
   ```
2. Open `eurosat_cnn_comparison.ipynb` in Jupyter, or upload it to Colab, and run all cells top to bottom. The dataset (EuroSAT, via Kaggle) downloads automatically in the setup cell.
3. GPU recommended — the notebook auto-detects CUDA and falls back to CPU otherwise.

## Results

| Model | Augmentation | Test Loss | Test Accuracy |
|---|---|---|---|
| TinyVGG (scratch) | No | 0.2717 | 0.9136 |
| TinyVGG (scratch) | Yes | 0.2106 | 0.9279 |
| ResNet18 (fine-tuned) | No | 0.0606 | 0.9812 |
| ResNet18 (fine-tuned) | Yes | 0.0595 | 0.9780 |

## Conclusion

Fine-tuned ResNet18 clearly outperforms TinyVGG trained from scratch, achieving both lower loss and higher accuracy across the board — a result of the rich, transferable features learned from ImageNet pretraining. TinyVGG benefits noticeably from data augmentation (better generalization, fewer misclassifications), while augmentation has a negligible, mixed effect on the already high-performing ResNet18. Overall, fine-tuning a pretrained model is by far the more effective strategy for this task, though a well-augmented from-scratch model can still close some of the gap.
