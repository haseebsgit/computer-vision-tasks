import cv2
import numpy as np
import os

# ------------------------------------
# 1. Motion detection using frame diff
# ------------------------------------
def get_motion_mask(prev, curr):
    diff = cv2.absdiff(prev, curr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    return mask

# ------------------------------------
# 2. White color detection
# ------------------------------------
def get_white_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask

# ------------------------------------
# MAIN WHITE CAR COUNTER
# ------------------------------------
def count_white_cars(video_path):

    if not os.path.exists(video_path):
        print("File not found")
        return

    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        print("Can't read video")
        return

    # Keep original resolution for processing
    prev = cv2.resize(prev, (160, 120))
    line_y = 80   # counting line
    offset = 6

    next_id = 0
    tracked = {}          # id → (cx, cy)
    counted = set()
    count = 1

    detections_for_draw = []   # store boxes for drawing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (160, 120))
        detections_for_draw = []

        # --- masks ---
        motion_mask = get_motion_mask(prev, frame)
        white_mask = get_white_mask(frame)
        final_mask = cv2.bitwise_and(motion_mask, white_mask)

        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            if cv2.contourArea(c) < 25:   # lowered for small cars
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append((x, y, w, h))
            detections_for_draw.append((x, y, w, h))

        # -------------------------------
        # Track objects (simple centroid)
        # -------------------------------
        new_tracked = {}

        for (x, y, w, h) in detections:
            cx = x + w//2
            cy = y + h//2

            assigned = False

            for oid, (px, py) in tracked.items():
                if abs(cx - px) < 30 and abs(cy - py) < 30:
                    new_tracked[oid] = (cx, cy)
                    assigned = True
                    break

            if not assigned:
                new_tracked[next_id] = (cx, cy)
                next_id += 1

        tracked = new_tracked

        # -------------------------------
        # Count cars crossing line
        # -------------------------------
        for oid, (cx, cy) in tracked.items():
            if (line_y - offset) < cy < (line_y + offset):
                if oid not in counted:
                    counted.add(oid)
                    count += 1
                    print(f"Car counted! Total = {count}")

        # ----------------------------------
        # Display (scaled up for visibility)
        # ----------------------------------
        display = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_NEAREST)

        # Draw boxes (scaled)
        sx = 480 / 160
        sy = 360 / 120

        for (x, y, w, h) in detections_for_draw:
            X = int(x * sx); Y = int(y * sy)
            W = int(w * sx); H = int(h * sy)
            cv2.rectangle(display, (X, Y), (X+W, Y+H), (0, 255, 0), 2)

        # Draw line
        cy = int(line_y * sy)
        cv2.line(display, (0, cy), (480, cy), (255, 0, 0), 2)

        cv2.putText(display, f"Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow("White Car Counter", display)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        prev = frame.copy()

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal Count =", count)


# Run
count_white_cars(r"D:\computer vision tasks\motion analysis\traffic.avi")
