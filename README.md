# Handwritten Text Transcription Script

This script uses the TrOCR model to transcribe handwritten text from images (e.g., doctor's notes, US Constitution) and optionally normalizes the text into plain English. It provides a simple GUI for selecting image files and outputs the transcription results to the console, with an option to save them to a text file for easy viewing in an editor like VS Code.

## Features
- Transcribes handwritten text from images using the TrOCR model (`microsoft/trocr-large-handwritten`).
- Attempts a lightweight OCR using `pytesseract` (if installed) before falling back to TrOCR.
- Preprocesses images with multiple strategies (standard, minimal, no binarization) to improve transcription accuracy.
- Optionally normalizes medical prescriptions or shorthand into plain English (e.g., "Tab." to "Tablet").
- Provides a visualization option to see preprocessing steps (with the `--visualize` flag).
- Outputs results to the console and allows saving to a file (e.g., `doctor notes_transcription.txt`).

## Requirements
- **Operating System:** Windows, macOS, or Linux
- **Python Version:** Python 3.6–3.10
- **Hardware:** A CPU is sufficient, but a GPU (CUDA-enabled) will speed up the TrOCR model. At least 4GB of RAM is recommended for model loading.
- **Internet Connection:** Required to download the TrOCR model (~2.2GB) on first run.
- **Dependencies:**
  - `torch` (PyTorch for model inference)
  - `transformers` (Hugging Face library for TrOCR)
  - `pillow` (PIL for image handling)
  - `opencv-python` (OpenCV for image preprocessing)
  - `numpy` (for array operations)
  - `matplotlib` (for visualization of preprocessing steps)
  - `tkinter` (for the GUI file picker; usually included with Python)
  - (Optional) `pytesseract` and Tesseract OCR (for lightweight OCR fallback)

## Installation
1. **Install Python:**
   - Ensure you have Python 3.6–3.10 installed. You can download it from [python.org](https://www.python.org/downloads/).
   - Verify your Python version by running:
     ```
     python --version
     ```

2. **Install Required Libraries:**
   - Open a terminal (e.g., PowerShell on Windows, Terminal on macOS/Linux) and run:
     ```
     pip install torch transformers pillow opencv-python numpy matplotlib
     ```
   - **Note:** `tkinter` is usually included with Python. On Linux, you may need to install it separately:
     ```
     sudo apt-get install python3-tk
     ```

3. **(Optional) Install `pytesseract` for Lightweight OCR:**
   - Install Tesseract OCR:
     - On Windows: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
     - On macOS: `brew install tesseract`
     - On Linux: `sudo apt-get install tesseract-ocr`
   - Install the Python package:
     ```
     pip install pytesseract
     ```
   - Ensure Tesseract is in your system PATH. On Windows, you may need to add it manually (e.g., `C:\Program Files\Tesseract-OCR`).

4. **Download the Script:**
   - Save the `handwritten_transcription.py` script to your desired directory (e.g., `C:\path\to\your\folder` on Windows).

## Usage
1. **Prepare Your Image:**
   - Ensure you have a handwritten image file (e.g., `doctor notes.jpg`) in a supported format (`.png`, `.jpg`, `.jpeg`, `.bmp`).
   - Place the image in the same directory as the script or note its full path.

2. **Run the Script:**
   - Open a terminal in the directory containing the script:
     ```
     cd C:\path\to\your\folder
     ```
   - Run the script:
     ```
     python handwritten_transcription.py
     ```
   - **Optional:** To visualize preprocessing steps, add the `--visualize` flag:
     ```
     python handwritten_transcription.py --visualize
     ```

3. **Interact with the Script:**
   - A GUI file picker will open. Click "Select Image File" and choose your image (e.g., `doctor notes.jpg`).
   - The script will attempt a quick OCR using `pytesseract` (if installed). If it fails, it will proceed with TrOCR.
   - You’ll be asked: `Do you want to continue with more accurate OCR using TrOCR? This requires downloading a ~2.2GB model. (y/n)`.
     - Type `y` to proceed with TrOCR (recommended for better accuracy).
     - Type `n` to stop after the quick OCR attempt.
   - The script will process the image and display the transcription results in the terminal:
     ```
     Original Transcription: for services.
     ```
   - You’ll be asked: `Do you want to save results to a text file? (y/n)`.
     - Type `y` to save the results to a file (e.g., `doctor notes_transcription.txt`) in the same directory as the image.
     - Type `n` to skip saving and only see the results in the terminal.

4. **View Results:**
   - **In the Terminal:** The results are printed under `TRANSCRIPTION RESULTS`.
   - **In a File (if saved):** If you chose to save the results, open the file (e.g., `C:/Users/melis/OneDrive/Masaüstü/doctor notes_transcription.txt`) in a text editor like VS Code to see:
     ```
     Original Transcription:
     for services.
     ```

## Example Output
```
Using device: cpu
Opening file picker to select an image...
Selected file:C:\path\to\your\folder/doctor notes.jpg
Attempting to process image: C:\path\to\your\folder doctor notes.jpg
Image file opened successfully with PIL.
Attempting lightweight processing first...
Using pytesseract for quick OCR...
Simple OCR failed: tesseract is not installed or it's not in your PATH. See README file for more information.
```
## If Something Goes Wrong
- **The Script Won’t Start:**
  - Make sure Python is installed and the version is between 3.6 and 3.10 (`python --version`).
  - Check that you installed all the tools with `pip install torch transformers pillow opencv-python numpy matplotlib`.
- **It Can’t Find Your Image:**
  - Double-check the image path. Make sure the file (e.g., `doctor notes.jpg`) is in the right folder.
- **It Says Tesseract Isn’t Installed:**
  - That’s okay! It means you didn’t install Tesseract (the quick method). Just type `y` to use TrOCR instead.
  - If you want to use Tesseract, follow the setup steps for it.
- **The Model Download Fails:**
  - Make sure you’re connected to the internet. The TrOCR model (2.2GB) needs to download the first time.
  - If it fails, try running the script again.
- **The File Picker Window Doesn’t Show Up:**
  - You might be missing `tkinter`. On Windows, it’s usually included with Python. If not, try reinstalling Python.
  - If it doesn’t work, the script will ask you to type the image path instead.
- **Still Having Problems?**
  - Look at the error message in the terminal. Copy it and ask for help (e.g., share it with a friend or online).

## Extra Tips
- If you want to see how the script prepares the image before reading it, run:
- python handwritten_transcription.py --visualize
- This will show pictures of the steps it takes to clean up the image.
- The script works best with clear handwriting. If the text is too messy, it might not read it perfectly.

## License
- This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
- Contributions are welcome! Please feel free to submit a Pull Request.
