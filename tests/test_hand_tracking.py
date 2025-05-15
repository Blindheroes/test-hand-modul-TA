"""
Utility script to test the hand tracking module directly
"""
from hand_tracking import HandTracker
import cv2
import time
import mediapipe as mp
import sys

# Add the parent directory to the path
sys.path.append("..")


def main():
    print("Starting hand tracking test...")

    # Initialize hand tracker
    tracker = HandTracker()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Process frame with hand tracker
        processed_frame = tracker.process_frame(
            frame)        # Check for gestures
        pointing = tracker.is_pointing()
        selecting = tracker.is_selecting()

        # Display gesture status
        cv2.putText(processed_frame, f"Pointing: {pointing}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if pointing else (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Selecting: {selecting}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if selecting else (0, 0, 255), 2)

        # Get pointer position
        pointer_pos = tracker.get_pointer_position()
        if pointer_pos:
            x, y = pointer_pos
            cv2.circle(processed_frame, (x, y), 10, (0, 255, 255), -1)

        # Show frame
        cv2.imshow("Hand Tracking Test", processed_frame)

        # Break loop if ESC key is pressed
        if cv2.waitKey(1) == 27:  # ESC key
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
