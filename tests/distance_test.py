"""
Distance Test for Hand Gesture Recognition
Tests hand gesture recognition accuracy at different distances
"""
import cv2
import time
import numpy as np
from .base_test import BaseTest


class DistanceTest(BaseTest):
    def __init__(self, camera=0, duration=20, results_dir="results"):
        super().__init__(camera, duration, results_dir)
        self.test_name = "distance"
        self.distances = ["close", "optimal", "far"]  # 1m, 2m, 3m
        self.gestures = ["pointing", "selecting"]
        self.current_gesture_index = 0

    def countdown(self, cap, message):
        """Display a 5-second countdown with a specific message"""
        print(f"{message} - Test will begin in 5 seconds...")

        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                continue

            # Calculate remaining time
            remaining = 5 - int(time.time() - start_time)

            # Process frame with hand tracker (just for display)
            processed_frame = self.hand_tracker.process_frame(frame)

            # Display countdown on frame
            self.display_status(processed_frame, message,
                                position=(50, 50), color=(0, 0, 255))
            self.display_status(processed_frame, f"Mulai dalam: {remaining} detik", position=(
                50, 100), color=(0, 0, 255))

            # Show the frame
            cv2.imshow("Distance Test", processed_frame)

            # Break loop if ESC key is pressed
            if cv2.waitKey(1) == 27:  # ESC key
                return False

        print("Mulai pengujian!")
        return True

    def run(self):
        """Run the distance test"""
        # Open camera
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera}")
            return

        results = []
        overall_start_time = time.time()
        test_phase = "setup"  # setup, gesture_selection, countdown, testing
        current_distance = None
        current_gesture = None
        start_test_time = None
        frame_count = 0

        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Process frame with hand tracker
            processed_frame = self.hand_tracker.process_frame(frame)
            frame_count += 1

            if test_phase == "setup":
                # Display instructions to select distance
                self.display_status(
                    processed_frame, "Pilih jarak pengujian:", position=(30, 30))
                self.display_status(
                    processed_frame, "1 = Dekat (1 meter)", position=(30, 70))
                self.display_status(
                    processed_frame, "2 = Optimal (2 meter)", position=(30, 110))
                self.display_status(
                    processed_frame, "3 = Jauh (3 meter)", position=(30, 150))
                self.display_status(
                    processed_frame, "ESC = Keluar", position=(30, 190))

                # Get user input for distance
                key = cv2.waitKey(1)
                if key == ord('1'):
                    current_distance = "close"
                    test_phase = "gesture_selection"
                elif key == ord('2'):
                    current_distance = "optimal"
                    test_phase = "gesture_selection"
                elif key == ord('3'):
                    current_distance = "far"
                    test_phase = "gesture_selection"
                elif key == 27:  # ESC
                    break

            elif test_phase == "gesture_selection":
                # Display instructions to select gesture
                self.display_status(
                    processed_frame, f"Jarak: {current_distance}", position=(30, 30))
                self.display_status(
                    processed_frame, "Pilih gerakan pengujian:", position=(30, 70))
                self.display_status(
                    processed_frame, "1 = Pointing (jari telunjuk)", position=(30, 110))
                self.display_status(
                    processed_frame, "2 = Selecting (telunjuk dan kelingking)", position=(30, 150))
                self.display_status(
                    processed_frame, "0 = Kembali pilih jarak", position=(30, 190))
                self.display_status(
                    processed_frame, "ESC = Keluar", position=(30, 230))

                # Get user input for gesture
                key = cv2.waitKey(1)
                if key == ord('1'):
                    current_gesture = "pointing"
                    test_phase = "countdown"
                elif key == ord('2'):
                    current_gesture = "selecting"
                    test_phase = "countdown"
                elif key == ord('0'):
                    current_distance = None
                    test_phase = "setup"
                elif key == 27:  # ESC
                    break

            elif test_phase == "countdown":
                message = f"Bersiap untuk pengujian jarak {current_distance} dengan gerakan {current_gesture}"

                # Show countdown
                cv2.imshow("Distance Test", processed_frame)
                if self.countdown(cap, message):
                    test_phase = "testing"
                    start_test_time = time.time()
                else:
                    # ESC pressed during countdown
                    break
                    
            elif test_phase == "testing":
                # Check time limit for this test
                elapsed_test_time = time.time() - start_test_time
                remaining_time = self.duration - elapsed_test_time

                if remaining_time <= 0:
                    # Return to gesture selection for the same distance
                    test_phase = "gesture_selection"
                    # Display a completion message for a moment
                    completion_start = time.time()
                    while time.time() - completion_start < 2:  # Show for 2 seconds
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        processed_frame = self.hand_tracker.process_frame(frame)
                        self.display_status(
                            processed_frame, f"Pengujian {current_gesture} pada jarak {current_distance} selesai!",
                            position=(30, 30), color=(0, 255, 0))
                        self.display_status(
                            processed_frame, "Kembali ke menu pemilihan gerakan...",
                            position=(30, 70), color=(0, 255, 0))
                        cv2.imshow("Distance Test", processed_frame)
                        if cv2.waitKey(1) == 27:
                            break
                    continue

                # Display current test information
                self.display_status(
                    processed_frame, f"Jarak: {current_distance}", position=(30, 30))
                self.display_status(
                    processed_frame, f"Gerakan: {current_gesture}", position=(30, 70))
                self.display_status(
                    processed_frame, f"Waktu tersisa: {int(remaining_time)}s", position=(30, 110))

                # Check for current gesture
                is_detected = False
                if current_gesture == "pointing":
                    is_detected = self.hand_tracker.is_pointing()
                elif current_gesture == "selecting":
                    is_detected = self.hand_tracker.is_selecting()
                
                # Record results for every frame (detected or not)
                results.append([
                    elapsed_test_time,       # Time since test started
                    frame_count,             # Frame number
                    current_distance,        # Current distance
                    current_gesture,         # Target gesture
                    1 if is_detected else 0  # 1 if detected, 0 if not
                ])

                # Display detection status
                if is_detected:
                    self.display_status(processed_frame, "Terdeteksi: Ya ", position=(
                        30, 150), color=(0, 255, 0))
                else:
                    self.display_status(processed_frame, "Terdeteksi: Tidak ", position=(
                        30, 150), color=(0, 0, 255))

                # Allow ESC to exit during testing
                if cv2.waitKey(1) == 27:  # ESC
                    break

            # Show frame
            cv2.imshow("Distance Test", processed_frame)
            
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        # Save results
        headers = ["timestamp", "frame_number",
                   "distance", "gesture", "detected"]
        self.save_results(results, headers)

        print(f"Pengujian jarak selesai. {len(results)} sampel data dicatat.")
