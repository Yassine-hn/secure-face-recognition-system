#!/usr/bin/env python3
"""Real-time debugging of mouth MAR values to identify the issue."""

import cv2
import mediapipe as mp
import numpy as np
from liveness.checks.mouth import mouth_aspect_ratio

# MediaPipe face mesh indices
MOUTH_MAR_INDICES = [61, 84, 17, 314, 405, 320, 307, 375]

def extract_landmarks_2d(landmarks, indices, w, h):
    """Extract 2D landmarks from MediaPipe face mesh."""
    points = []
    for idx in indices:
        if idx < len(landmarks.landmark):
            x = landmarks.landmark[idx].x * w
            y = landmarks.landmark[idx].y * h
            points.append([x, y])
    return points

def main():
    print("Mouth MAR Real-time Debug")
    print("Keep your mouth CLOSED and observe the MAR values.")
    print("Press 'q' to quit, 's' to save current frame info")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. Using simulated data...")
        simulate_mouth_mar()
        return
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    frame_count = 0
    mar_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            pts = extract_landmarks_2d(face_landmarks, MOUTH_MAR_INDICES, w, h)
            
            if len(pts) == 8:
                mar = mouth_aspect_ratio(pts)
                mar_history.append(mar)
                
                # Keep only last 30 values
                if len(mar_history) > 30:
                    mar_history.pop(0)
                
                # Draw the mouth points
                for i, (x, y) in enumerate(pts):
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    cv2.putText(frame, str(MOUTH_MAR_INDICES[i]), (int(x)-15, int(y)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
                # Draw connections to visualize the mouth shape
                for i in range(len(pts)):
                    next_i = (i + 1) % len(pts)
                    cv2.line(frame, (int(pts[i][0]), int(pts[i][1])), 
                            (int(pts[next_i][0]), int(pts[next_i][1])), 
                            (255, 255, 0), 1)
                
                # Display MAR value and statistics
                avg_mar = np.mean(mar_history) if mar_history else 0
                min_mar = np.min(mar_history) if mar_history else 0
                max_mar = np.max(mar_history) if mar_history else 0
                
                cv2.putText(frame, f'MAR: {mar:.4f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Avg: {avg_mar:.4f}', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Min: {min_mar:.4f}', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Max: {max_mar:.4f}', (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show threshold indicators
                cv2.putText(frame, f'Threshold 0.4: {"PASS" if mar >= 0.4 else "FAIL"}', 
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if mar >= 0.4 else (0, 0, 255), 2)
                
                # Show if this would pass with relative threshold
                if len(mar_history) >= 5:
                    baseline = np.mean(mar_history[:5])  # First 5 frames as baseline
                    current_increase = mar - baseline
                    cv2.putText(frame, f'Baseline: {baseline:.4f}', (10, 190),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f'Increase: {current_increase:.4f}', (10, 220),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if current_increase >= 0.2 else (0, 0, 255), 2)
                    cv2.putText(frame, f'Rel 0.2: {"PASS" if current_increase >= 0.2 else "FAIL"}', 
                               (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 0) if current_increase >= 0.2 else (0, 0, 255), 2)
        
        cv2.imshow('Mouth MAR Debug', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if mar_history:
                print(f"\nFrame {frame_count} - MAR Statistics:")
                print(f"  Current: {mar:.4f}")
                print(f"  Average: {np.mean(mar_history):.4f}")
                print(f"  Range: {np.min(mar_history):.4f} - {np.max(mar_history):.4f}")
                print(f"  Points: {pts}")
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    
    if mar_history:
        print("\nFinal Statistics:")
        print(f"  Average MAR: {np.mean(mar_history):.4f}")
        print(f"  Min MAR: {np.min(mar_history):.4f}")
        print(f"  Max MAR: {np.max(mar_history):.4f}")
        print(f"  Std Dev: {np.std(mar_history):.4f}")

def simulate_mouth_mar():
    """Simulate MAR values when camera is not available."""
    print("\nSimulating mouth MAR values...")
    
    # Simulate realistic MAR values for a closed mouth with some noise
    base_mar = 0.07  # Natural closed mouth
    noise_std = 0.02  # Small variations
    
    print(f"Simulated MAR values (base={base_mar:.4f}, noise_std={noise_std:.4f}):")
    
    for i in range(20):
        mar = base_mar + np.random.normal(0, noise_std)
        mar = max(0, mar)  # Ensure non-negative
        print(f"  Frame {i+1}: MAR = {mar:.4f} (>=0.4: {mar >= 0.4})")
    
    print(f"\nWith relative threshold (baseline={base_mar:.4f}):")
    for i in range(20):
        mar = base_mar + np.random.normal(0, noise_std)
        mar = max(0, mar)
        increase = mar - base_mar
        print(f"  Frame {i+1}: increase = {increase:.4f} (>=0.2: {increase >= 0.2})")

if __name__ == "__main__":
    main()
