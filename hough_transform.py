import cv2
import numpy as np
import math

def detect_lines_and_calculate_angle(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Create a copy for drawing results
    result_image = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None:
        print("No lines detected! Try adjusting the Hough Transform parameters.")
        return None
    
    # Convert lines to more usable format and draw them
    detected_lines = []
    
    for i, line in enumerate(lines[:2]):  # Get only the first two strongest lines
        rho, theta = line[0]
        
        # Convert polar coordinates to Cartesian
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # Calculate two points for the line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        detected_lines.append((rho, theta, (x1, y1, x2, y2)))
        
        # Draw the line
        color = (0, 0, 255) if i == 0 else (255, 0, 0)  # Red and Blue
        cv2.line(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Add line label
        cv2.putText(result_image, f'Line {i+1}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Calculate angle between the two lines
    if len(detected_lines) >= 2:
        theta1 = detected_lines[0][1]  # Angle of first line
        theta2 = detected_lines[1][1]  # Angle of second line
        
        # Calculate the acute angle between lines (in degrees)
        angle_rad = abs(theta1 - theta2)
        angle_deg = math.degrees(min(angle_rad, np.pi - angle_rad))
        
        # Display the angle on the image
        cv2.putText(result_image, f'Angle: {angle_deg:.2f}°', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        print(f"Line 1 angle: {math.degrees(theta1):.2f}°")
        print(f"Line 2 angle: {math.degrees(theta2):.2f}°")
        print(f"Angle between lines: {angle_deg:.2f}°")
    
    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Edge Detection', edges)
    cv2.imshow('Detected Lines with Angle', result_image)
    
    # Wait for key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return angle_deg if len(detected_lines) >= 2 else None

# Usage with your image
image_path = "picture3.jpg"  # Replace with your image path
angle = detect_lines_and_calculate_angle(image_path)

def detect_lines_probabilistic(image_path):
    """Alternative method using probabilistic Hough Transform"""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    result_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=30, maxLineGap=10)
    
    if lines is None:
        print("No lines detected!")
        return None
    
    # Calculate angles for all detected lines
    line_angles = []
    for i, line in enumerate(lines[:2]):  # Use first two lines
        x1, y1, x2, y2 = line[0]
        
        # Draw the line
        color = (0, 0, 255) if i == 0 else (255, 0, 0)
        cv2.line(result_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result_image, f'Line {i+1}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Calculate angle
        angle = math.atan2(y2 - y1, x2 - x1)
        line_angles.append(angle)
    
    # Calculate angle between lines
    if len(line_angles) >= 2:
        angle_rad = abs(line_angles[0] - line_angles[1])
        angle_deg = math.degrees(min(angle_rad, np.pi - angle_rad))
        
        cv2.putText(result_image, f'Angle: {angle_deg:.2f}°', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        print(f"Angle between lines: {angle_deg:.2f}°")
    
    # Display results
    cv2.imshow('Probabilistic Hough Lines', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return angle_deg if len(line_angles) >= 2 else None

# Usage
# angle = detect_lines_probabilistic("your_image.jpg")