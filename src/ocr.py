import os
import tempfile

import easyocr
import re
import cv2

import numpy as np

reader = easyocr.Reader(['en'])
def extract_text(image):
    texts = []

    # pass 1: raw
    texts.append(" ".join(reader.readtext(image, detail=0)))

    # chọn text tốt nhất
    text = max(texts, key=len)

    return text