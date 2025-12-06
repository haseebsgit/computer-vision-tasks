import cv2
import numpy as np


video_path = r'D:\computer vision tasks\motion analysis\traffic.avi'
resize_height = 800       
min_area = 1200       



pixel_is_black_threshold = 65  


required_black_percentage = 0.78

trafficObj = cv2.VideoCapture(video_path)

if not trafficObj.isOpened():
    print("Error: Could not open video.")
    exit()

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
se5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print(f"Tracking only if more than {int(required_black_percentage*100)}% of the body is black...")
print("Press 'ESC' to exit.")

while True:
    ret, frame = trafficObj.read()
    if not ret:
        break

    # Resize
    height, width = frame.shape[:2]
    ratio = resize_height / height
    new_width = int(width * ratio)
    frame = cv2.resize(frame, (new_width, resize_height))

    # Motion Detection
    fg_mask = bg_subtractor.apply(frame)
    fBW = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, se5)
    fBW = cv2.morphologyEx(fBW, cv2.MORPH_CLOSE, se5)

    contours, _ = cv2.findContours(fBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 1. Tighter Crop (Focus on the Door/Side Panel)
            roi_y_start = y + int(h * 0.3) # Skip top 30%
            roi_y_end   = y + int(h * 0.7) # Skip bottom 30%
            roi_x_start = x + int(w * 0.2) # Skip side edges
            roi_x_end   = x + int(w * 0.8)
            
            if roi_y_end > roi_y_start and roi_x_end > roi_x_start:
                car_body = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                gray_body = cv2.cvtColor(car_body, cv2.COLOR_BGR2GRAY)
                
                # 2. Count Black Pixels
                black_pixels_count = np.sum(gray_body < pixel_is_black_threshold)
                total_pixels = gray_body.size
                
                # 3. Calculate Percentage
                black_ratio = black_pixels_count / total_pixels

                # 4. Strict Logic: Don't track if below 50%
                if black_ratio >= required_black_percentage:
                    # Draw Box (Red)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
                    # Label with percentage
                    label = f"Dark Car: {int(black_ratio*100)}%"
                    cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Task 3: Track Dark Cars (Strict 50%)', frame)
    
    if cv2.waitKey(30) & 0xFF == 27:
        break

trafficObj.release()
cv2.destroyAllWindows()