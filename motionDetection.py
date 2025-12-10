#please run this in google colab
# ----------------------------
# IMPORTS
# ----------------------------
from IPython.display import display, Javascript, Image, clear_output
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np
import time

# ----------------------------
# REQUEST WEBCAM PERMISSION
# ----------------------------
display(Javascript("""
navigator.mediaDevices.getUserMedia({video: true})
  .then(stream => {
    console.log('Webcam access granted');
    stream.getTracks().forEach(track => track.stop());
  })
  .catch(err => console.log('Webcam permission denied', err));
"""))
time.sleep(2)  # Wait for permission to register

# ----------------------------
# FUNCTION TO CAPTURE A FRAME
# ----------------------------
def capture_frame():
    js = Javascript('''
    async function captureFrame() {
        const video = document.createElement('video');
        const stream = await navigator.mediaDevices.getUserMedia({video:true});
        video.srcObject = stream;
        await video.play();
        await new Promise(resolve => setTimeout(resolve, 200)); // Small delay

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        stream.getVideoTracks()[0].stop();
        return canvas.toDataURL('image/jpeg', 0.8);
    }
    ''')
    display(js)

    for attempt in range(2):
        try:
            data = eval_js("captureFrame()")
            binary = b64decode(data.split(',')[1])
            img_array = np.frombuffer(binary, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed, retrying...", e)
            time.sleep(1)
    return None

# ----------------------------
# MOTION DETECTION LOOP
# ----------------------------
print("🚀 Starting Motion Detection...\nLook at the camera! Capturing frames...")

previous_gray = None
motion_count = 0  # Total frames with motion

for i in range(15):  # Capture 15 frames
    frame = capture_frame()
    if frame is None:
        print("⚠️ No frame captured!")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Set baseline for first frame
    if previous_gray is None:
        previous_gray = gray
        continue

    # Compute difference between frames
    diff = cv2.absdiff(previous_gray, gray)

    # Threshold differences
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    # Draw boxes around motion
    for c in contours:
        if cv2.contourArea(c) < 800:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    previous_gray = gray

    clear_output(wait=True)
    display(Image(data=cv2.imencode('.jpg', frame)[1].tobytes()))

    if motion_detected:
        motion_count += 1
        print(f"⚠️ Motion detected in frame {i+1}!")
    else:
        print(f"✅ No motion detected in frame {i+1}.")

    time.sleep(0.2)

print(f"✅ Motion detection loop finished! Total frames with motion: {motion_count}")
