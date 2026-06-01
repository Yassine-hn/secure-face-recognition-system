#!/usr/bin/env python3
"""Debug script to check typical MAR values from MediaPipe face landmarks."""

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe face mesh indices for mouth (8 points)
MOUTH_MAR_INDICES = [78, 191, 80, 13, 308, 415, 310, 14]

def extract_landmarks_2d(landmarks, indices, w, h):
    """Extract 2D landmarks from MediaPipe face mesh."""
    points = []
    for idx in indices:
        if idx < len(landmarks.landmark):
            x = landmarks.landmark[idx].x * w
            y = landmarks.landmark[idx].y * h
            points.append((x, y))
    return points

def mouth_aspect_ratio(points):
    """Compute MAR using the standard 8-point mouth landmark layout."""
    landmarks = np.asarray(list(points), dtype=np.float32)
    a = np.linalg.norm(landmarks[2] - landmarks[6])
    b = np.linalg.norm(landmarks[3] - landmarks[5])
    c = np.linalg.norm(landmarks[0] - landmarks[4])
    if c == 0.0:
        return 0.0
    return float((a + b) / (2.0 * c))

def main():
    # Try to capture from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. Using sample data instead.")
        
        # Sample data for a closed mouth (approximate positions)
        sample_points = [
            (100, 150),  # left corner
            (120, 145),  # left top
            (140, 143),  # top left
            (160, 143),  # top right
            (180, 150),  # right corner
            (160, 157),  # bottom right
            (140, 159),  # bottom left
            (120, 157),  # left bottom
        ]
        mar = mouth_aspect_ratio(sample_points)
        print(f"Sample closed mouth MAR: {mar}")
        print(f"Threshold 0.5 would pass: {mar >= 0.5}")
        print(f"Threshold 0.4 would pass: {mar >= 0.4}")
        print(f"Threshold 0.3 would pass: {mar >= 0.3}")
        return
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    print("Press 'q' to quit. Keep mouth closed and observe MAR values.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            pts = extract_landmarks_2d(face_landmarks, MOUTH_MAR_INDICES, w, h)
            
            if len(pts) == 8:
                mar = mouth_aspect_ratio(pts)
                
                # Draw the mouth points
                for i, (x, y) in enumerate(pts):
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    cv2.putText(frame, str(i), (int(x)-5, int(y)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Display MAR value
                cv2.putText(frame, f'MAR: {mar:.3f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show threshold comparisons
                cv2.putText(frame, f'>=0.3: {mar >= 0.3}', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if mar >= 0.3 else (0, 0, 255), 2)
                cv2.putText(frame, f'>=0.4: {mar >= 0.4}', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if mar >= 0.4 else (0, 0, 255), 2)
                cv2.putText(frame, f'>=0.5: {mar >= 0.5}', (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if mar >= 0.5 else (0, 0, 255), 2)
        
        cv2.imshow('Mouth MAR Debug', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()
