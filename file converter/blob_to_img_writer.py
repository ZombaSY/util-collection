import base64
import cv2
import numpy as np

input_name = 'temp.bin'
output_name = 'temp.jpg'

with open(input_name, 'rb') as f:
    f = f.read()
    img = base64.standard_b64decode(f)
    img = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), -1)
    cv2.imwrite(output_name, img)
