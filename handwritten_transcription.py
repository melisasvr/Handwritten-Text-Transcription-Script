import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
import warnings
import logging

# Disable progress bars from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configure logging to replace the progress bars with cleaner output
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Monkey patch tqdm to disable progress bars
import transformers
transformers.utils.logging.is_progress_bar_enabled = lambda: False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple OCR fallback function (was missing in the original code)
def simple_ocr_fallback(image_path):
    """Use pytesseract for quick OCR when available"""
    try:
        import pytesseract
        from PIL import Image
        
        # Open image with PIL to avoid encoding issues
        img = Image.open(image_path)
        
        # Apply basic preprocessing to improve OCR
        img = img.convert('L')  # Convert to grayscale
        
        # Use pytesseract for OCR
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Simple OCR failed: {e}")
        return "Simple OCR failed. Proceeding to advanced method."

# Attempt to verify model is already downloaded or handle gracefully
def check_model_availability():
    """Check if model files already exist locally"""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_dir = os.path.join(cache_dir, "models--microsoft--trocr-large-handwritten")
    
    if os.path.exists(model_dir):
        snapshots = [d for d in os.listdir(model_dir) if d.startswith("snapshots")]
        if snapshots:
            snapshot_dir = os.path.join(model_dir, snapshots[0])
            model_bin = os.path.join(snapshot_dir, "pytorch_model.bin")
            if os.path.exists(model_bin) and os.path.getsize(model_bin) > 2_000_000_000:  # > 2GB
                return True
    
    print("Model needs to be downloaded (~2.2GB). This may take some time.")
    print("If download fails, try running the script again or check your internet connection.")
    return False

# Preprocessing function for handwritten images with visualization option
def preprocess_image(image_path, visualize=False):
    # Read image with PIL first to handle encoding issues
    try:
        pil_img = Image.open(image_path)
        # Convert PIL image to OpenCV format
        img_np = np.array(pil_img)
        if len(img_np.shape) == 3:
            original_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            original_img = img_np
    except Exception as e:
        print(f"Error opening image with PIL: {e}")
        # Fallback to direct OpenCV reading
        original_img = cv2.imread(image_path)
    
    if original_img is None:
        raise FileNotFoundError(f"Could not open or find the image: {image_path}")
    
    # Convert to grayscale if it's not already
    if len(original_img.shape) == 3:
        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        img = original_img.copy()
    
    # Enhance contrast
    img_contrast = cv2.equalizeHist(img)
    
    # Denoise
    img_denoised = cv2.fastNlMeansDenoising(img_contrast, h=10)
    
    # Binarize (adaptive thresholding for varied handwriting)
    img_binary = cv2.adaptiveThreshold(
        img_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Resize while maintaining aspect ratio (TrOCR expects reasonable sizes)
    target_height = 384
    aspect_ratio = img_binary.shape[1] / img_binary.shape[0]
    target_width = int(target_height * aspect_ratio)
    img_resized = cv2.resize(img_binary, (target_width, target_height))
    
    # Convert to RGB (TrOCR expects 3 channels)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    # Visualize preprocessing steps if requested
    if visualize:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 2)
        plt.title("Grayscale")
        plt.imshow(img, cmap='gray')
        
        plt.subplot(2, 3, 3)
        plt.title("Enhanced Contrast")
        plt.imshow(img_contrast, cmap='gray')
        
        plt.subplot(2, 3, 4)
        plt.title("Denoised")
        plt.imshow(img_denoised, cmap='gray')
        
        plt.subplot(2, 3, 5)
        plt.title("Binary")
        plt.imshow(img_binary, cmap='gray')
        
        plt.subplot(2, 3, 6)
        plt.title("Resized")
        plt.imshow(img_rgb)
        
        plt.tight_layout()
        plt.show()
    
    return img_rgb

# Function to recognize handwriting based on optimal preprocessing strategy
def recognize_handwriting(image_path, processor, model, visualize=False):
    # Try multiple preprocessing approaches
    preprocessing_methods = {
        "standard": lambda: preprocess_image(image_path, visualize),
        "minimal": lambda: cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR),
        "no_binarize": lambda: process_without_binarize(image_path)
    }
    
    best_text = ""
    best_confidence = -1
    best_method = ""
    
    print("Trying different preprocessing methods...")
    
    for method_name, method_func in preprocessing_methods.items():
        try:
            print(f"Testing {method_name} preprocessing...")
            img = method_func()
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img)
            
            # Process image for TrOCR
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            
            # Decode to text
            transcribed_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Calculate a simple heuristic confidence score (length of non-space characters)
            confidence = len(transcribed_text.replace(" ", ""))
            
            print(f"  Result: '{transcribed_text}' (confidence: {confidence})")
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_text = transcribed_text
                best_method = method_name
        
        except Exception as e:
            print(f"  Error with {method_name} method: {str(e)}")
    
    print(f"Selected best method: {best_method} with confidence {best_confidence}")
    return best_text

# Alternative preprocessing method without binarization
def process_without_binarize(image_path):
    # Open with PIL first to handle encoding issues
    pil_img = Image.open(image_path)
    img = np.array(pil_img)
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img = cv2.equalizeHist(img)
    img = cv2.fastNlMeansDenoising(img, h=10)
    
    # Skip binarization
    
    # Resize
    target_height = 384
    aspect_ratio = img.shape[1] / img.shape[0]
    target_width = int(target_height * aspect_ratio)
    img = cv2.resize(img, (target_width, target_height))
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

# Translate to plain English using rule-based approach instead of model
def translate_to_plain_english(text):
    """Use simple rules to normalize text instead of using T5 model"""
    try:
        # Simple normalization rules
        result = text
        
        # Remove extra spaces
        result = ' '.join(result.split())
        
        # Fix common OCR errors
        result = result.replace("l -", "1-")
        result = result.replace("l-", "1-")
        result = result.replace("O-", "0-")
        result = result.replace("o-", "0-")
        
        # Fix common prescription abbreviations
        result = result.replace("Tab.", "Tablet")
        result = result.replace("Tab ", "Tablet ")
        result = result.replace("Rx", "Prescription")
        result = result.replace("rx", "prescription")
        result = result.replace("sig:", "Instructions:")
        
        return result
    except Exception as e:
        print(f"Error during translation to plain English: {e}")
        return text  # Return original text if translation fails

# Function to get image path (prompt or file picker)
def get_image_path():
    if len(sys.argv) > 1 and sys.argv[1] != "--visualize":
        return sys.argv[1]  # Use command line argument if provided
    
    print("Opening file picker to select an image...")
    try:
        # Make a more visible root window for the file picker
        root = tk.Tk()
        root.title("Select Handwritten Image")
        root.geometry("400x200")  # Set window size
        
        # Create a label with instructions
        label = tk.Label(root, text="Click the button below to open file picker")
        label.pack(pady=20)
        
        # Variable to store selected path
        selected_path = [None]
        
        # Function to handle file selection
        def select_file():
            file_path = filedialog.askopenfilename(
                parent=root,
                title="Select Handwritten Image File",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
            )
            if file_path:  # If a file was selected
                selected_path[0] = file_path
                success_label = tk.Label(root, text=f"Selected: {os.path.basename(file_path)}", fg="green")
                success_label.pack(pady=10)
                # Close the window after a short delay
                root.after(1500, root.destroy)
            else:
                error_label = tk.Label(root, text="No file selected!", fg="red")
                error_label.pack(pady=10)
        
        # Button to trigger file selection
        select_button = tk.Button(root, text="Select Image File", command=select_file)
        select_button.pack(pady=20)
        
        # Start the GUI event loop
        root.mainloop()
        
        # Check if a file was selected
        if selected_path[0]:
            print(f"Selected file: {selected_path[0]}")
            return selected_path[0]
        else:
            # If no file was selected through GUI, fall back to command line input
            print("No file selected. Please enter the path manually:")
            return input().strip()
    
    except Exception as e:
        print(f"Error with file picker: {e}")
        print("Falling back to manual input. Please enter the full path to your handwritten image:")
        return input().strip()

# Main function to process an uploaded image
def process_handwritten_image(image_path, visualize=False):
    try:
        # Check if model is available
        check_model_availability()
        
        # Load TrOCR model and processor
        print("Loading TrOCR model...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device)
        
        # Recognize handwriting
        print("Transcribing handwriting...")
        transcribed_text = recognize_handwriting(image_path, processor, model, visualize)
        print(f"Transcribed text: {transcribed_text}")
        
        # Translate to plain English if needed
        if should_translate(transcribed_text):
            print("Normalizing text...")
            plain_text = translate_to_plain_english(transcribed_text)
            print(f"Normalized text: {plain_text}")
            return transcribed_text, plain_text
        else:
            print("Text appears to be in plain English already, skipping normalization.")
            return transcribed_text, transcribed_text
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        # For debugging
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", f"Error: {str(e)}"

# Determine if text needs translation
def should_translate(text):
    # Simplified heuristic: if text is short enough and not just simple instructions
    if len(text) < 10:
        return False
    
    # Check if it appears to be a medical prescription or simple form
    medical_terms = ['tab', 'mg', 'capsule', 'prescription', 'rx', 'dose']
    if any(term in text.lower() for term in medical_terms):
        return True
    
    return False

# Main function
def main():
    # Determine if we should visualize the preprocessing steps
    visualize = "--visualize" in sys.argv
    
    # Get image path from user
    image_path = get_image_path()
    
    if not image_path:
        print("Error: No image path provided.")
        return
    
    # Clean up the path (sometimes quotation marks can be included)
    image_path = image_path.strip('"\'')
    
    print(f"Attempting to process image: {image_path}")
    
    # Test if file exists using a more robust method
    try:
        Image.open(image_path)
        print("Image file opened successfully with PIL.")
    except Exception as e:
        print(f"Error: Image file not found or cannot be opened: '{image_path}'")
        print(f"Error details: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print("Please provide a valid image path.")
        return
    
    # Start with simple processing that doesn't require downloading large models
    try:
        print("\nAttempting lightweight processing first...")
            
        # First try simple OCR if available (doesn't require large downloads)
        try:
            import pytesseract
            print("Using pytesseract for quick OCR...")
            simple_text = simple_ocr_fallback(image_path)
            print("\n" + "="*50)
            print("QUICK OCR RESULT:")
            print("="*50)
            print(simple_text)
            print("="*50 + "\n")
            
            print("Do you want to continue with more accurate OCR using TrOCR? This requires downloading a ~2.2GB model. (y/n)")
            response = input().strip().lower()
            if response != 'y':
                print("Quick OCR process completed. No additional processing performed.")
                return
        except ImportError:
            print("pytesseract not available, proceeding with TrOCR...")
        
        # Process with TrOCR model
        transcribed_text, normalized_text = process_handwritten_image(image_path, visualize)
        
        # Print results
        print("\nResults:")
        print(f"Original Transcription: {transcribed_text}")
        if transcribed_text != normalized_text:
            print(f"Normalized Text: {normalized_text}")
        
        # Display results directly in console instead of saving to external file
        print("\n" + "="*50)
        print("TRANSCRIPTION RESULTS:")
        print("="*50)
        print(f"Original Transcription: {transcribed_text}")
        if transcribed_text != normalized_text:
            print(f"Normalized Text: {normalized_text}")
        print("="*50)
        
        # Ask if user wants to save results to file
        print("\nDo you want to save results to a text file? (y/n)")
        save_response = input().strip().lower()
        if save_response == 'y':
            output_file = os.path.splitext(image_path)[0] + "_transcription.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Original Transcription:\n{transcribed_text}\n\n")
                if transcribed_text != normalized_text:
                    f.write(f"Normalized Text:\n{normalized_text}")
            print(f"Results saved to: {output_file}")
        else:
            print("Results not saved to file.")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()