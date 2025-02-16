import os
from PIL import Image
import pytesseract

folder_path = 'D:/aitech/test1'
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img)
        print(f'Extracted text from {filename}:\n{text}')