# IMDB Neural Sentiment Analyzer (Flet Edition)

A modern, autonomous Natural Language Processing desktop application built with **Python**, **PyTorch**, and the **Flet** UI framework. 

This application analyzes the sentiment of any movie or TV show review, predicting whether the text is positive or negative using a custom Feedforward Neural Network trained on 25,000 real-world IMDB reviews.

## Why This Version is Superior

This iteration represents the ultimate, production-ready version of the project. It improves upon standard terminal-based ML scripts in several key ways:

1. **The "All-in-One" Autonomous Boot:** Unlike traditional ML projects that require you to manually run a "train" script and then an "inference" script, this application is completely self-healing. If it detects that the neural weights are missing, it will automatically download the Stanford IMDB dataset, train the AI in the background, and then seamlessly launch the GUI.
2. **Instant State Persistence:** By serializing both the PyTorch tensor weights (`model.pth`) and the custom 10,000-word JSON dictionary (`words.json`), the application achieves zero-overhead instant booting on all subsequent runs.
3. **Modern Graphical Interface:** Replaces clunky terminal inputs with a sleek, dark-mode desktop GUI. It features multiline text ingestion, interactive buttons with ripple effects, and dynamic color-coded visual feedback.
4. **Zero-Disk I/O Pipeline:** The training sequence extracts and reads 25,000 text files directly into RAM from a compressed `.tar.gz` archive, bypassing hard-drive I/O bottlenecks entirely.

## Neural Architecture

The engine underneath the UI utilizes a highly optimized NLP pipeline:
* **Text Normalization:** Custom Regex filtering to strip punctuation and HTML tags.
* **Embedding Layer:** Converts raw word IDs into 32-dimensional dense mathematical vectors.
* **Mean Pooling:** Averages the maximum 200 word-vectors across the sequence dimension to capture the overall context of the review.
* **Linear Classifier:** Compresses the context into a single node, squashed by a Sigmoid activation function to provide a 0.0 to 1.0 confidence score.

## Tech Stack

* **Language:** Python 3.8+
* **Frontend GUI:** Flet (v0.80.0+)
* **Machine Learning:** PyTorch (`torch`, `torch.nn`, `torch.optim`)
* **Standard Libraries:** `re`, `tarfile`, `urllib`, `json`, `collections`

## How to Run

1. **Install Requirements:**
   Ensure you have the necessary libraries installed:
   ```bash
   pip install torch flet
