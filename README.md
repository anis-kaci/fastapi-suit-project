# Multimodal Interview Analysis Platform Backend API

This project provides a robust and high-performance backend API, built with **FastAPI**, designed to power a multimodal interview analysis application. It specializes in processing both **audio transcription** and **facial emotion analysis** from video inputs, enabling comprehensive insights into human interactions during interviews.

## Overview

The API acts as the core engine, receiving media (audio/video) from a frontend application and leveraging powerful open-source machine learning models to extract valuable data. This data includes spoken content (transcription), as well as real-time and summary insights into facial expressions and sentiment.

## Features

* **Comprehensive Audio Transcription:**
    * **Pre-recorded Video/Audio:** Converts spoken language from entire video or audio files into accurate text transcripts.
    * **Real-time Streaming:** Processes live audio streams (e.g., from a webcam microphone) to provide near-instantaneous transcription updates, crucial for live interview scenarios.

* **Advanced Facial Emotion & Sentiment Analysis:**
    * **Frame-by-Frame Detection:** Identifies faces in video frames (both pre-recorded and live webcam streams).
    * **Emotion Prediction:** Predicts a range of discrete emotions (e.g., happy, sad, angry, neutral, surprise, fear, disgust) with confidence scores for each detected face.
    * **Sentiment Inference:** Derives overall sentiment (positive, negative, neutral) from facial expressions.
    * **Bounding Box Overlays:** Provides coordinates for drawing bounding boxes around detected faces on the frontend, visually highlighting the analysis.

* **Text Sentiment & Emotion Analysis (NLP):**
    * Performs Natural Language Processing (NLP) on the transcribed text to determine the overall sentiment and potentially extract textual emotions (e.g., if the language used suggests anger or joy).

* **Data Persistence & Session Management:**
    * API endpoints for saving detailed analysis results (transcriptions, facial emotion timelines, text sentiment summaries) linked to specific interview sessions.

## Technologies Used

* **FastAPI:** The chosen framework for building the API. Its asynchronous capabilities (ASGI), automatic data validation with Pydantic, and built-in interactive API documentation (Swagger UI/OpenAPI) ensure high performance, reliability, and ease of use.
* **Uvicorn:** The lightning-fast ASGI server that runs the FastAPI application.
* **Whisper (OpenAI):** A state-of-the-art model for Automatic Speech Recognition (ASR), utilized for highly accurate audio transcription.
* **TensorFlow / Keras:** The deep learning framework responsible for loading and executing the custom-trained Convolutional Neural Network (CNN) model for facial emotion analysis. This ensures efficient inference on detected faces.
* **OpenCV (`cv2`):** A foundational library for computer vision, essential for tasks like video frame extraction, image decoding, grayscale conversion, and robust face detection within the analysis pipeline.
* **`python-multipart`:** Enables FastAPI to efficiently handle large file uploads, particularly for video files.
* **`numpy`:** Provides fundamental numerical computing capabilities, crucial for data manipulation within the machine learning pipelines.
* **`tempfile`:** Used for secure and efficient management of temporary audio and image files during processing.
* **`websockets`:** The underlying library enabling the real-time, bidirectional communication required for live webcam analysis.

## Core API Endpoints

The API exposes several key endpoints to facilitate the multimodal analysis:

* **`POST /analyze-interview-video`**:
    * **Description:** Processes a complete uploaded video file.
    * **Input:** Video file (e.g., `.mp4`, `.webm`) via `FormData`.
    * **Output:** Comprehensive JSON object containing the full transcription, a detailed timeline of facial emotion and sentiment detections (bounding boxes, probabilities) per second, and overall text sentiment/emotion analysis.

* **`WebSocket /ws/analyze-webcam`**:
    * **Description:** Establishes a real-time, bidirectional connection for live webcam analysis.
    * **Input (from Frontend):** JPEG image frames (binary blobs) captured from the user's webcam.
    * **Output (to Frontend):** Real-time JSON updates with detected face bounding boxes, emotion probabilities, and sentiment scores for each frame.

* **`POST /transcribe_real_time`**:
    * **Description:** Handles real-time audio chunks for transcription during live webcam sessions.
    * **Input:** Small audio segments (e.g., `audio/webm;codecs=opus`) from the microphone via `FormData`.
    * **Output:** Incremental text transcription for the received audio chunk.

* **`POST /analyze-text-sentiment`**:
    * **Description:** Performs sentiment analysis on a given text input.
    * **Input:** Raw text string.
    * **Output:** JSON containing the overall sentiment label (positive, negative, neutral) and associated confidence scores.

* **`POST /analyze-text-emotion`**:
    * **Description:** Classifies emotions present in a given text input.
    * **Input:** Raw text string.
    * **Output:** JSON containing probability scores for various emotions detected in the text.

* **Data Persistence Endpoints (e.g., `/api/save-transcription`, `/api/save-facial-emotions`, `/api/save-text-sentiment`, `/api/save-webcam-analysis`):**
    * **Description:** Dedicated endpoints for saving the processed analysis results into the backend database, associated with a unique session ID.
    * **Input:** Structured JSON payloads containing the respective analysis data (transcriptions, aggregated facial emotions, text analysis summaries).
    * **Output:** Confirmation of successful data persistence.

## Setup and Installation

Follow these steps to set up and run the API on your local machine.

### Prerequisites

* **Python (3.9 to 3.12 recommended):** TensorFlow currently does not support Python 3.13. It's crucial to use a compatible Python version for TensorFlow dependencies.
* **FFmpeg:** Essential for Whisper to process audio files.

#### Installing FFmpeg (Windows)

1.  Download a pre-built static version of FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) (e.g., from gyan.dev or BtbN).
2.  Extract the downloaded ZIP file to a stable, easily accessible location (e.g., `C:\ffmpeg`).
3.  Add the `bin` directory of your FFmpeg installation to your system's `PATH` environment variable (e.g., `C:\ffmpeg\bin`). This allows the system to find `ffmpeg.exe` from any directory.
4.  **Restart your terminal/command prompt** after modifying the `PATH` variable for the changes to take effect. Verify the installation by typing `ffmpeg -version` in your terminal.

### Project Setup

1.  **Clone the repository (if applicable) or navigate to your project directory:**

    ```bash
    cd /path/to/your/fastapi-suit-project
    ```

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage project dependencies in isolation and avoid conflicts with other Python projects.

    ```bash
    # Ensure you are using a compatible Python version (e.g., 3.12)
    # If you have multiple Python versions, you might need to specify the executable:
    # "C:\Python312\python.exe" -m venv .venv  (on Windows)
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**

    * **On Windows (Command Prompt/PowerShell):**

        ```bash
        .venv\Scripts\activate
        ```

    * **On macOS/Linux (Bash/Zsh):**

        ```bash
        source .venv/bin/activate
        ```

    Your terminal prompt should now display `(.venv)` at the beginning, indicating the virtual environment is active.

4.  **Install Project Dependencies:**
    All required Python packages are listed in `reqs.txt`.

    ```bash
    pip install -r reqs.txt
    ```

    * **Note on TensorFlow:** If you encounter `No matching distribution found for tensorflow`, it's most likely due to an incompatible Python version. Please ensure your Python version is within the recommended range (3.9-3.12).
    * **Note on `openai-whisper`:** The Whisper model (defaulting to the `base` model size) will be automatically downloaded the first time it is used by the application. This might take a moment.

5.  **Place Your Facial Analysis Model:**
    Ensure your pre-trained CNN model file (e.g., `best_model_so_far.keras`) is located in the root directory of your project (the same directory as `main.py`). If you place it elsewhere, remember to update the `EMOTION_MODEL_PATH` variable within your `main.py` file to its correct relative or absolute path.

## Running the API

Once all dependencies are installed and your virtual environment is active, you can start the API server:

```bash
uvicorn main:app --reload