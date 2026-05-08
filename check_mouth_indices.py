#!/usr/bin/env python3
"""Check if mouth landmark indices are correct for MediaPipe Face Mesh."""

import cv2
import mediapipe as mp

# MediaPipe Face Mesh - Mouth outer contour indices
# According to MediaPipe documentation, the mouth outer contour is:
# 61, 84, 17, 314, 405, 320, 307, 375 (but these are for lips)
# The indices used in the code might be for different mouth points

CURRENT_INDICES = [78, 191, 80, 13, 308, 415, 310, 14]

# Alternative mouth indices from MediaPipe documentation
LIP_OUTER_TOP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIP_OUTER_BOTTOM = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Let's use the standard 8-point mouth configuration
# These are commonly used for MAR calculation
MOUTH_8_POINTS = [61, 84, 17, 314, 405, 320, 307, 375]

def draw_landmarks(image, landmarks, indices, color=(0, 255, 0), label=""):
    """Draw specific landmarks on image."""
    h, w = image.shape[:2]
    for i, idx in enumerate(indices):
        if idx < len(landmarks.landmark):
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            cv2.circle(image, (x, y), 3, color, -1)
            cv2.putText(image, str(idx), (x-10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Draw label
    cv2.putText(image, label, (10, 30 + indices[0] * 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    print("Showing different mouth landmark indices.")
    print("Green: Current indices [78, 191, 80, 13, 308, 415, 310, 14]")
    print("Blue: Alternative 8-point [61, 84, 17, 314, 405, 320, 307, 375]")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2 RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Draw current indices in green
            draw_landmarks(frame, face_landmarks, CURRENT_INDICES, 
                          (0, 255, 0), "Current")
            
            # Draw alternative indices in blue
            draw_landmarks(frame, face_landmarks, MOUTH_8_POINTS, 
                          (255, 0, 0), "Alternative")
        
        cv2.imshow('Mouth Landmark Comparison', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()
