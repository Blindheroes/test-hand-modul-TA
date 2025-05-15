# Demo script to test Hand Tracking module with toggleable features
import cv2
import os
import sys
import time
from hand_tracking import HandTracker


def main():
    # Initialize hand tracker with both features enabled by default
    tracker = HandTracker(use_threshold_adaptation=True,
                          use_temporal_filtering=True)

    # Start video capture
    cap = cv2.VideoCapture(0)

    # Flag to track current feature state
    features_on = True
    last_toggle_time = time.time()

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)

        # Process the frame to detect hands
        frame = tracker.process_frame(frame)

        # Visualize finger states and gestures
        frame = tracker.visualize_finger_states(frame)

        # Add instructions
        cv2.putText(frame, "Press 'A' to toggle Threshold Adaptation",
                    (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'T' to toggle Temporal Filtering",
                    (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'B' to toggle both features",
                    (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Press 'Q' to quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display the frame
        cv2.imshow('Hand Tracking Demo', frame)

        # Handle key presses with a cooldown to prevent multiple toggles
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()

        if current_time - last_toggle_time > 0.5:  # 500ms cooldown for toggles
            if key == ord('a'):
                # Toggle threshold adaptation
                tracker.toggle_features(
                    use_threshold_adaptation=not tracker.use_threshold_adaptation)
                last_toggle_time = current_time
                print(
                    f"Threshold Adaptation: {'ON' if tracker.use_threshold_adaptation else 'OFF'}")

            elif key == ord('t'):
                # Toggle temporal filtering
                tracker.toggle_features(
                    use_temporal_filtering=not tracker.use_temporal_filtering)
                last_toggle_time = current_time
                print(
                    f"Temporal Filtering: {'ON' if tracker.use_temporal_filtering else 'OFF'}")

            elif key == ord('b'):
                # Toggle both features
                features_on = not features_on
                tracker.toggle_features(use_threshold_adaptation=features_on,
                                        use_temporal_filtering=features_on)
                last_toggle_time = current_time
                print(f"All features: {'ON' if features_on else 'OFF'}")

            elif key == ord('q'):
                # Quit
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
