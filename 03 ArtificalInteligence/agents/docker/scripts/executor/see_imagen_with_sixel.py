"""
sudo apt update
sudo apt install mlterm
"""

import sys
import io
from sixel import SixelWriter
import cv2

# Si no pasas argumentos, usa una imagen por defecto o error
if len(sys.argv) < 2:
    print("Uso: python see_ascii.py <imagen>")
    sys.exit(1)

plot = cv2.imread(sys.argv[1])
if plot is None:
    print(
        f"Error: Could not load image '{sys.argv[1]}'. Please check the path and file integrity."
    )
    sys.exit(1)

# Results image as bytes
im_bytes = cv2.imencode(
    ".png",
    plot,
)[1].tobytes()

# Image bytes as a file-like object
mem_file = io.BytesIO(im_bytes)

# Create sixel writer object
w = SixelWriter()

# Draw the sixel image in the terminal
w.draw(mem_file)
