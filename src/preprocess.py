import cv2

def preprocess_image(image_path):
    # đọc ảnh
    img = cv2.imread(image_path)

    # upscale (quan trọng nhất)
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    img = gray_scale_image(img)

    img = enhance_contrast(img)

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def gray_scale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def enhance_contrast(gray):
    return cv2.convertScaleAbs(gray, alpha=1.5, beta=30)