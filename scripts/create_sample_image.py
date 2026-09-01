import os
import cv2
import numpy as np

def main():
    os.makedirs("data/sample/image", exist_ok=True)
    
    # 1. Create a skin-like background image (BGR format)
    # Skin tone is typically around: B=140, G=160, R=220
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = [140, 160, 220]
    
    # 2. Draw a shaded circle representing ankle swelling
    cv2.circle(img, (150, 150), 45, (100, 110, 160), -1)
    
    # 3. Add random noise texture to pass blur and contrast checks
    noise = np.random.randint(0, 60, (300, 300, 3), dtype=np.int16)
    img_noise = np.clip(img.astype(np.int16) + noise - 30, 0, 255).astype(np.uint8)
    
    # 4. Draw bounding border and some contrast text
    cv2.rectangle(img_noise, (10, 10), (290, 290), (0, 0, 0), 2)
    cv2.putText(img_noise, "AI-QTriage Football Injury", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    
    output_path = "data/sample/image/football_injury.jpg"
    cv2.imwrite(output_path, img_noise)
    print(f"Created sample injury image at: {output_path}")

if __name__ == "__main__":
    main()
