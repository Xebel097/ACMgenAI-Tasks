# Task 1: 🎙️ India's Got Latent — AI Stage

An interactive Streamlit web application powered by **LangChain** and **Groq (Llama-3.3-70B)**. Select an act persona—ranging from a comedy roast bot to an Elizabethan playwright—and banter with an AI contestant live on stage with full conversation memory.

---

## 📋 Features

* **Multiple Stage Personas:** Switch seamlessly between unique AI act personas (RoastBot, ShakespeareBot, Emoji Translator, Strict Hostel Warden).
* **Stateful Chat Memory:** Retains conversation history throughout the session using LangChain's `InMemoryChatMessageHistory` and `RunnableWithMessageHistory`.
* **Zero API Prompting:** The Groq API key is pre-configured directly inside the script for immediate execution.
* **Fast Inference:** Powered by Groq's low-latency API running `llama-3.3-70b-versatile`.

---

## 🛠️ Prerequisites

Make sure you have the following installed on your system:

* **Python 3.9** or higher
* **pip** (Python package installer)

---

## 🚀 Setup & Installation

### 1. Download or Clone the Repository

Clone this repository or download the source code files into a local folder:

```bash
git clone [https://github.com/Xebel097/India-s-got-Latent-chatbot.git](https://github.com/Xebel097/India-s-got-Latent-chatbot.git)
cd India-s-got-Latent-chatbot
```
### 2. Set Up a Virtual Environment (Recommended)
Create and activate a virtual environment to manage dependencies:

On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
On Windows:
```DOS
python -m venv venv
venv\Scripts\activate
```
### 3. Install Required Dependencies
Install the necessary Python packages using pip:
```bash
pip install streamlit langchain-groq langchain-core python-dotenv
```
## 🔑 API Key Configuration
The API key to run this chatbot can be pre-defined in the .env file. The user is advised to utilise their personal key (generate one from https://console.groq.com/keys) for the project but in absolute necessetiy.

## 🏃 Running the App
Run the application using Streamlit:

```bash
streamlit run Task-1.py
```
Streamlit will automatically open the app in your default browser at http://localhost:8501.

## 📁 Project Structure

|__ Task-1.py          # Main Streamlit application

|__ Task1 requirements.txt # All the dependencies one requires

|__ Task1.env.example # .env example file containing the sensitive keys

|__ .gitignore  # contains the .env file

|__ README.md    # Project setup and documentation

<p align="center">
  <img src="https://github.com/Xebel097/ACMgenAI-Tasks/blob/main/Task1_in_action.png" alt="Alt Text" width="4000" />
</p>

# Task 2: TinyVGG-vs-ResNet18

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

## Model Architectures
 
### TinyVGG (trained from scratch)
 
A small VGG-style CNN with no pretrained weights — everything is learned from the EuroSAT data alone. Input: 64×64×3, `hidden=32`.
 
| Block | Layers |
|---|---|
| Block 1 | Conv(3→32, 3×3) → ReLU → Conv(32→32, 3×3) → ReLU → MaxPool(2×2) |
| Block 2 | Conv(32→64, 3×3) → ReLU → Conv(64→64, 3×3) → ReLU → MaxPool(2×2) |
| Block 3 | Conv(64→128, 3×3) → ReLU → MaxPool(2×2) |
| Classifier | Flatten → Dropout(0.3) → Linear(→10 classes) |
 
Each block doubles the channel depth while halving spatial size via max-pooling — a classic VGG pattern (stack small 3×3 convs, then downsample). After 3 pooling stages, a 64×64 image shrinks to 8×8, giving a flattened feature vector of `8 × 8 × 128 = 8192`, which feeds into a single linear layer for the 10-class output. Dropout before the final layer is the only regularization beyond data augmentation.
 
This model acts as the "control" — no ImageNet priors, no transfer learning — showing what's achievable learning purely from EuroSAT's own signal.
 
### ResNet18 (fine-tuned, pretrained)
 
A standard ResNet18, initialized with ImageNet-pretrained weights, then adapted to this task.
 
- The full ResNet18 backbone (4 stages of residual blocks with skip connections, which let gradients bypass layers and allow much deeper training without degradation) is kept largely intact.
- Only the final fully-connected layer is replaced:
```python
  model.fc = nn.Linear(in_feats, num_classes)  # in_feats=512 → 10
```
- Input is resized to 224×224 (ResNet's native ImageNet input size) and normalized with ImageNet mean/std, since the backbone's learned filters expect that same input distribution it was trained on.
- The whole network is fine-tuned end-to-end (not frozen) — but starting from weights that already encode general visual features (edges, textures, shapes) rather than random initialization.
This model wins not just because it's deeper, but because its pretrained filters already generalize well to visual patterns beyond ImageNet's specific classes (including satellite imagery textures) — reflected in the ~98% vs ~91-93% test accuracy gap over TinyVGG.
## Results

| Model | Augmentation | Test Loss | Test Accuracy |
|---|---|---|---|
| TinyVGG (scratch) | No | 0.2717 | 0.9136 |
| TinyVGG (scratch) | Yes | 0.2106 | 0.9279 |
| ResNet18 (fine-tuned) | No | 0.0606 | 0.9812 |
| ResNet18 (fine-tuned) | Yes | 0.0595 | 0.9780 |

## Conclusion

Fine-tuned ResNet18 clearly outperforms TinyVGG trained from scratch, achieving both lower loss and higher accuracy across the board — a result of the rich, transferable features learned from ImageNet pretraining. TinyVGG benefits noticeably from data augmentation (better generalization, fewer misclassifications), while augmentation has a negligible, mixed effect on the already high-performing ResNet18. Overall, fine-tuning a pretrained model is by far the more effective strategy for this task, though a well-augmented from-scratch model can still close some of the gap.
