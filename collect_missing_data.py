import os
import cv2
import time
import mediapipe as mp

# The dataset we downloaded has A-Z and 1-9. It is only missing '0'.
MISSING_CLASSES = ['0']
FRAMES_PER_CLASS = 150
SAVE_DIR = "dataset/split_dataset/train"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    print("=========================================================")
    print(" ISL Smart Data Collector (With Validation)")
    print("=========================================================")
    print("Order of signs you will be asked for:")
    print(" -> " + ", ".join(MISSING_CLASSES))
    print("=========================================================\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize MediaPipe Hands for validation
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
    
        for cls in MISSING_CLASSES:
            cls_dir = os.path.join(SAVE_DIR, cls)
            os.makedirs(cls_dir, exist_ok=True)
            
            existing = len([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])
            if existing >= FRAMES_PER_CLASS:
                print(f"Skipping '{cls}', already has {existing} valid images.")
                continue
                
            print(f"\n---> NEXT SIGN: '{cls}'")
            print("Get your hand in position.")
            
            # Wait for 'c' to start capturing
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                
                cv2.putText(frame, f"Sign: {cls} | Press 'c' to start", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("Smart Data Collector", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Start validated capturing
            count = existing
            while count < FRAMES_PER_CLASS:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                
                # VALIDATION: Check if MediaPipe can clearly see the hand
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    # Hand detected! Valid image.
                    # Draw skeleton on display frame for user feedback
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Save a center crop of the RAW frame (without drawn lines)
                    h, w = frame.shape[:2]
                    size = min(h, w)
                    cy, cx = h//2, w//2
                    crop = frame[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
                    crop = cv2.resize(crop, (300, 300))
                    
                    img_path = os.path.join(cls_dir, f"{cls}_{int(time.time()*1000)}_{count}.jpg")
                    cv2.imwrite(img_path, crop)
                    
                    count += 1
                    cv2.putText(display_frame, f"Saved: {count}/{FRAMES_PER_CLASS}", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    # Invalid image (blurry, hand out of frame, etc.)
                    cv2.putText(display_frame, "VALIDATION FAILED: Show hand clearly", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                cv2.imshow("Smart Data Collector", display_frame)
                cv2.waitKey(50) # Small delay
                
            print(f"Finished validated capture for '{cls}'!")

    cap.release()
    cv2.destroyAllWindows()
    print("\nAll missing data collected and validated successfully!")

if __name__ == "__main__":
    main()
