import cv2
import numpy as np

def check_image_quality(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"status": "error", "reason": "Không đọc được ảnh"}

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    issues = []

    # 1. Resolution
    if w < 300 or h < 300:
        issues.append("resolution_too_low")

    # 2. Blur — dùng local variance thay vì global
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = lap.var()
    local_sharpness = np.mean(np.abs(lap))
    is_blurry = blur_score < 100
    if local_sharpness < 10:
        issues.append("low_text_sharpness")

    # 3. Brightness global
    brightness = np.mean(gray)
    if brightness < 50:
        issues.append("too_dark")
    elif brightness > 220:
        issues.append("too_bright")

    # 4. Uneven lighting — chia ảnh thành grid 3x3, so sánh từng vùng
    grid_means = []
    for i in range(3):
        for j in range(3):
            cell = gray[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            grid_means.append(np.mean(cell))
    lighting_variance = np.std(grid_means)
    if lighting_variance > 40:
        issues.append("uneven_lighting")

    # 5. Vùng tối quá nhiều
    dark_ratio = np.sum(gray < 50) / gray.size
    if dark_ratio > 0.3:
        issues.append("too_many_dark_regions")

    # 6. Contrast local (không phải global std)
    contrast = gray.std()
    if contrast < 20:
        issues.append("low_contrast")

    # 7. Skew
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                             minLineLength=100, maxLineGap=10)
    skew_angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:
                angles.append(angle)
        if angles:
            skew_angle = float(np.median(angles))
            if abs(skew_angle) > 5:
                issues.append("skewed")

    # 8. Status — phân 3 mức
    critical = {"resolution_too_low", "too_dark"}
    warning = {
        "low_text_sharpness",
        "uneven_lighting",
        "low_contrast",
        "too_many_dark_regions",
        "too_bright",
        "skewed"
    }
    if any(i in critical for i in issues):
        status = "bad"
    elif any(i in warning for i in issues):
        status = "warning"
    else:
        status = "ok"

    quality = {
        "status": status,
        "issues": issues,
        "metrics": {
            "resolution": f"{w}x{h}",
            "blur_score": round(blur_score, 1),
            "local_sharpness": round(local_sharpness, 2),
            "is_blurry": is_blurry,
            "brightness": round(float(brightness), 1),
            "dark_ratio": round(float(dark_ratio), 3),
            "lighting_variance": round(float(lighting_variance), 1),
            "contrast": round(float(contrast), 1),
            "skew_angle": round(skew_angle, 2),
        }
    }

    # Log
    status_icon = {"ok": "✅", "warning": "⚠️", "bad": "❌"}[status]
    print(f"{status_icon} Chất lượng ảnh: {status.upper()}")
    for k, v in quality["metrics"].items():
        print(f"   {k:<20}: {v}")
    if issues:
        print(f"   Issues: {', '.join(issues)}")

    return quality