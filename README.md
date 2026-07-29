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
The API key to run this chatbot is pre-defined in the .env.example file. The user is advised to utilise their personal key (generate one from https://console.groq.com/keys) for the project but in absolute necessetiy, one could utilise the pre-defined Groq API key.

## 🏃 Running the App
Run the application using Streamlit:

```bash
streamlit run app.py
```
Streamlit will automatically open the app in your default browser at http://localhost:8501.

## 📁 Project Structure

├── app.py          # Main Streamlit application
└── README.md       # Project setup and documentation

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

## Results

| Model | Augmentation | Test Loss | Test Accuracy |
|---|---|---|---|
| TinyVGG (scratch) | No | 0.2717 | 0.9136 |
| TinyVGG (scratch) | Yes | 0.2106 | 0.9279 |
| ResNet18 (fine-tuned) | No | 0.0606 | 0.9812 |
| ResNet18 (fine-tuned) | Yes | 0.0595 | 0.9780 |

## Conclusion

Fine-tuned ResNet18 clearly outperforms TinyVGG trained from scratch, achieving both lower loss and higher accuracy across the board — a result of the rich, transferable features learned from ImageNet pretraining. TinyVGG benefits noticeably from data augmentation (better generalization, fewer misclassifications), while augmentation has a negligible, mixed effect on the already high-performing ResNet18. Overall, fine-tuning a pretrained model is by far the more effective strategy for this task, though a well-augmented from-scratch model can still close some of the gap.
