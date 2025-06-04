# fastapi-suit-project
# Multimodal Chatbot API

This project provides a FastAPI-based backend API for processing multimodal inputs, specifically focusing on **audio transcription** and **facial emotion analysis**. It leverages powerful open-source machine learning models to offer these capabilities as accessible RESTful endpoints.

## Features

* **Audio Transcription:** Convert spoken language from audio files into text using the local Whisper model.

* **Facial Emotion Analysis:** Detect faces in images and predict their associated emotions (e.g., happy, sad, angry, neutral) using a custom-trained Convolutional Neural Network (CNN) model.

## Technologies Used

* **FastAPI:** A modern, fast (high-performance) web framework for building APIs with Python 3.7+.

* **Uvicorn:** An ASGI server for running FastAPI applications.

* **Whisper (OpenAI):** A robust model for automatic speech recognition (ASR), used for audio transcription.

* **TensorFlow / Keras:** Deep learning framework used to load and run the custom CNN model for facial emotion analysis.

* **OpenCV (`cv2`):** Library for computer vision tasks, used for image processing (decoding, grayscale conversion, face detection) in the facial analysis pipeline.

* **`python-multipart`:** Required for handling file uploads in FastAPI.

* **`numpy`:** Fundamental package for numerical computing in Python.

* **`tempfile`:** For secure handling of temporary files during audio and image processing.

## Setup and Installation

Follow these steps to set up and run the API on your local machine.

### Prerequisites

* **Python (3.9 to 3.12 recommended):** TensorFlow currently does not support Python 3.13. It's crucial to use a compatible Python version.

* **FFmpeg:** Required by Whisper for audio processing.

#### Installing FFmpeg (Windows)

1.  Download a pre-built static version of FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) (e.g., from gyan.dev or BtbN).

2.  Extract the downloaded ZIP file to a stable location (e.g., `C:\ffmpeg`).

3.  Add the `bin` directory of your FFmpeg installation to your system's `PATH` environment variable (e.g., `C:\ffmpeg\bin`).

4.  **Restart your terminal/command prompt** after adding to PATH. Verify by typing `ffmpeg -version`.

### Project Setup

1.  **Clone the repository (if applicable) or navigate to your project directory:**

    ```bash
    cd /path/to/your/fastapi-suit-project
    ```

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies in isolation.

    ```bash
    # Ensure you are using a compatible Python version (e.g., 3.12)
    # If you have multiple Python versions, specify the executable:
    # "C:\Python312\python.exe" -m venv .venv
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**

    * **On Windows (Command Prompt):**

        ```bash
        .venv\Scripts\activate
        ```

    * **On macOS/Linux:**

        ```bash
        source .venv/bin/activate
        ```

    Your terminal prompt should now show `(.venv)` at the beginning.

4.  **Install Project Dependencies:**
    
    ```bash
    pip install -r reqs.txt
    ```

    * **Note on TensorFlow:** If you encounter `No matching distribution found for tensorflow`, it's likely due to an incompatible Python version. Ensure you are using Python 3.9-3.12.

    * **Note on `openai-whisper`:** This will download the Whisper model (`base` by default) on first use.

5.  **Place Your Facial Analysis Model:**
    Ensure your pre-trained CNN model file (`best_model_so_far.keras`) is located in the root directory of your project (the same directory as `main.py`), or update the `EMOTION_MODEL_PATH` variable in `main.py` to its correct path.

## Running the API

Once all dependencies are installed and your virtual environment is active:

```bash
uvicorn main:app --reload