import cv2
import pytesseract
from PIL import Image

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image to get a binary image
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresholded

def ocr(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Convert OpenCV image to PIL format
    pil_image = Image.fromarray(processed_image)
    
    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(pil_image)
    
    return text

# Example usage
image_path = "2.jpg"
text = ocr(image_path)
print("Extracted text:", text)
