import cv2
import pytesseract
import numpy as np
import re

# Function to evaluate the mathematical expressions
def evaluate_expression(expression):
    try:
        # Replace the 'x' with '*' to handle multiplication
        expression = expression.replace('x', '*')
        result = eval(expression)
        # Convert to integer if the result is a whole number
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception as e:
        print(f"Error evaluating expression {expression}: {e}")
        return ""

# Load the input image
image_path = r'C:\Users\praka\Desktop\Machine Learning\Homework6\HW6-Q2.png'
image = cv2.imread(image_path)

# Preprocess the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
_, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
processed_image = cv2.erode(threshold_image, kernel, iterations=1)

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Use pytesseract to perform OCR and extract text from the preprocessed image
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(processed_image, config=custom_config)

line_coordinates = [(262, 129), (260, 254), (265, 404)]

# Process the extracted text to form valid mathematical expressions
expressions = re.findall(r'(\d+\s*[\+\-\*/x]\s*\d+)', text)

# Calculate and annotate each expression in the image
output_image = image.copy()
data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)

for i, expr in enumerate(expressions):
    result = evaluate_expression(expr)
    if result:
        line_x, line_y = line_coordinates[i % len(line_coordinates)]  
        result_x = line_x  # Align with the line's X-coordinate
        result_y = line_y  # Adjust vertically at the line

        # Overlay the result text at the specified coordinate
        cv2.putText(output_image, result, (result_x , result_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
         # Print detected expression and its evaluated result in console
        print(f"Detected Expression: {expr}, Result: {result}")

# Save the result
output_image_path = r'C:\Users\praka\Desktop\Machine Learning\Homework6\HW6-Q2output.png'
cv2.imwrite(output_image_path, output_image)

# Optionally display the image with results overlaid
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
