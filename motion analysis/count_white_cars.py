import cv2, math, os

class Tracker:
    def __init__(self):
        self.centers = {}  # id -> (x,y)
        self.disappeared = {} # id -> frames_lost
        self.next_id = 0

    def update(self, rects):
        objects_ids = []
        input_cents = [((2*x+w)//2, (2*y+h)//2) for (x,y,w,h) in rects]

        if not self.centers:
            for i, rect in enumerate(rects):
                self._register(input_cents[i], rect, objects_ids)
        else:
            obj_ids = list(self.centers.keys())
            obj_cents = list(self.centers.values())
            used_rows, used_cols = set(), set()

            for i, ic in enumerate(input_cents):
                min_dist, match_idx = float('inf'), -1
                for j, oc in enumerate(obj_cents):
                    if j in used_cols: continue
                    dist = math.hypot(ic[0]-oc[0], ic[1]-oc[1])
                    if dist < 80 and dist < min_dist: min_dist, match_idx = dist, j
                
                if match_idx != -1:
                    oid = obj_ids[match_idx]
                    self.centers[oid] = ic
                    self.disappeared[oid] = 0
                    objects_ids.append([*rects[i], oid])
                    used_rows.add(i); used_cols.add(match_idx)

            for i in range(len(input_cents)):
                if i not in used_rows: self._register(input_cents[i], rects[i], objects_ids)

            for i, oid in enumerate(obj_ids):
                if i not in used_cols:
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > 10:
                        del self.centers[oid]; del self.disappeared[oid]
        return objects_ids

    def _register(self, cent, rect, obj_ids):
        self.centers[self.next_id] = cent
        self.disappeared[self.next_id] = 0
        obj_ids.append([*rect, self.next_id])
        self.next_id += 1

def count_white_cars(path):
    if not os.path.exists(path): return print("File not found.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return print("Error opening video.")
    
    # Get dimensions to calculate resize height maintaining aspect ratio
    ret, frame = cap.read()
    if not ret: return print("Could not read video.")
    h_tgt = int(frame.shape[0] * (640 / frame.shape[1]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    tracker = Tracker()
    line_y, offset = int(h_tgt * 0.65), 12
    count, counted = 0, set()
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, h_tgt))
        
        # Process: Gray -> Threshold -> Morph -> Contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect and Track
        rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 400]
        boxes_ids = tracker.update(rects)
        
        # Draw and Count
        cv2.line(frame, (0, line_y), (640, line_y), (255, 0, 0), 2)
        for x, y, w, h, id in boxes_ids:
            cy = (2*y + h) // 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(id), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            
            if (line_y - offset) < cy < (line_y + offset) and id not in counted:
                count += 1
                counted.add(id)
                print(f"Car {id} Detected! Total: {count}")

        cv2.putText(frame, f"Count: {count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final Count: {count}")

if __name__ == "__main__":
    count_white_cars(r'D:\computer vision tasks\motion analysis\traffic.avi')