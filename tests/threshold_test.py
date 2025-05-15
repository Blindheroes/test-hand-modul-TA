# filepath: c:\Users\HP\OneDrive - Institut Teknologi Sepuluh Nopember\00 ITS\TA\Final-virtual-tryon-game (testing)\tests\threshold_test.py
"""
Threshold Adaptation Test for Hand Gesture Recognition
Tests the effectiveness of threshold adaptation based on distance for hand gesture recognition
"""
import cv2
import time
import numpy as np
from collections import deque
from .base_test import BaseTest


class ThresholdTest(BaseTest):
    def __init__(self, camera=0, duration=20, results_dir="results"):
        super().__init__(camera, duration, results_dir)
        self.test_name = "threshold"
        self.threshold_modes = ["adaptive", "fixed"]
        self.current_mode = None  # Will be selected during test
        self.distances = {
            "1": "close",      # 1 meter
            "2": "optimal",    # 2 meters
            "3": "far"         # 3 meters
        }
        # Will be selected by user during test
        self.current_distance = None

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
            cv2.imshow("Threshold Adaptation Test", processed_frame)

            # Break loop if ESC key is pressed
            if cv2.waitKey(1) == 27:  # ESC key
                return False

        print("Mulai pengujian!")
        return True

    def run(self):
        """Run the threshold adaptation test"""
        # Open camera
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera}")
            return

        results = []
        overall_start_time = time.time()
        # distance_selection, countdown_adaptive, testing_adaptive, countdown_fixed, testing_fixed
        test_phase = "distance_selection"
        start_test_time = None
        frame_count = 0

        # Store original threshold values
        original_threshold = self.hand_tracker.gesture_threshold

        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Process frame with hand tracker
            processed_frame = self.hand_tracker.process_frame(frame)
            frame_count += 1

            if test_phase == "distance_selection":
                # Display instructions
                self.display_status(
                    processed_frame, "Pengujian Adaptasi Threshold", position=(30, 30))
                self.display_status(
                    processed_frame, "Pilih jarak pengujian:", position=(30, 70))
                self.display_status(
                    processed_frame, "1 = Jarak Dekat (1 meter)", position=(30, 110))
                self.display_status(
                    processed_frame, "2 = Jarak Optimal (2 meter)", position=(30, 150))
                self.display_status(
                    processed_frame, "3 = Jarak Jauh (3 meter)", position=(30, 190))
                self.display_status(
                    processed_frame, "ESC = Keluar", position=(30, 230))

                # Get user input for distance selection
                key = cv2.waitKey(1)
                if key in [ord("1"), ord("2"), ord("3")]:
                    distance_key = chr(key)
                    self.current_distance = self.distances[distance_key]
                    test_phase = "countdown_adaptive"
                elif key == 27:  # ESC
                    break

            elif test_phase == "countdown_adaptive":
                message = f"Bersiap untuk pengujian dengan threshold adaptif pada jarak {self.current_distance}"

                # Show countdown
                cv2.imshow("Threshold Adaptation Test", processed_frame)
                if self.countdown(cap, message):
                    test_phase = "testing_adaptive"
                    start_test_time = time.time()
                    self.current_mode = "adaptive"

                    # Enable adaptive threshold based on distance
                    if self.current_distance == "close":
                        # Lower threshold for close distance (easier to detect)
                        self.hand_tracker.gesture_threshold = original_threshold * 0.7
                    elif self.current_distance == "optimal":
                        # Keep original threshold for optimal distance
                        self.hand_tracker.gesture_threshold = original_threshold
                    elif self.current_distance == "far":
                        # Higher threshold for far distance (harder to detect)
                        self.hand_tracker.gesture_threshold = original_threshold * 1.3
                else:
                    # ESC pressed during countdown
                    break

            elif test_phase == "testing_adaptive":
                # Check time limit for this test (10 seconds)
                elapsed_test_time = time.time() - start_test_time
                remaining_time = 10 - elapsed_test_time

                if remaining_time <= 0:
                    # Switch to testing with fixed threshold
                    test_phase = "countdown_fixed"
                    # Display a completion message for a moment
                    completion_start = time.time()
                    while time.time() - completion_start < 2:  # Show for 2 seconds
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        processed_frame = self.hand_tracker.process_frame(
                            frame)
                        self.display_status(
                            processed_frame, "Pengujian dengan threshold adaptif selesai!",
                            position=(30, 30), color=(0, 255, 0))
                        self.display_status(
                            processed_frame, "Bersiap untuk pengujian dengan threshold tetap...",
                            position=(30, 70), color=(0, 255, 0))
                        cv2.imshow("Threshold Adaptation Test",
                                   processed_frame)
                        if cv2.waitKey(1) == 27:
                            break
                    continue

                # Display current test information
                self.display_status(
                    processed_frame, f"Mode: Threshold Adaptif - Jarak: {self.current_distance}", position=(30, 30))
                self.display_status(
                    processed_frame, f"Waktu tersisa: {int(remaining_time)}s", position=(30, 70))
                self.display_status(
                    processed_frame, "Lakukan gerakan pointing dan selecting", position=(30, 110))
                self.display_status(
                    processed_frame, f"Nilai threshold saat ini: {self.hand_tracker.gesture_threshold:.2f}", position=(30, 150))

                # Check for gestures
                pointing_detected = self.hand_tracker.is_pointing()
                selecting_detected = self.hand_tracker.is_selecting()

                # Get the detected gesture name
                detected_gesture = None
                if pointing_detected:
                    detected_gesture = "pointing"
                elif selecting_detected:
                    detected_gesture = "selecting"

                # Record results for every frame
                results.append([
                    elapsed_test_time,          # Time since test started
                    frame_count,                # Frame number
                    self.current_mode,          # Adaptive or fixed threshold
                    self.current_distance,      # Current distance
                    self.hand_tracker.gesture_threshold,  # Current threshold value
                    detected_gesture,           # Detected gesture (or None)
                    1 if detected_gesture else 0  # 1 if detected, 0 if not
                ])

                # Display detection status
                if detected_gesture:
                    self.display_status(processed_frame, f"Terdeteksi: {detected_gesture}", position=(
                        30, 190), color=(0, 255, 0))
                else:
                    self.display_status(processed_frame, "Terdeteksi: Tidak", position=(
                        30, 190), color=(0, 0, 255))

                # Allow ESC to exit during testing
                if cv2.waitKey(1) == 27:  # ESC
                    break

            elif test_phase == "countdown_fixed":
                message = f"Bersiap untuk pengujian dengan threshold tetap pada jarak {self.current_distance}"

                # Show countdown
                cv2.imshow("Threshold Adaptation Test", processed_frame)
                if self.countdown(cap, message):
                    test_phase = "testing_fixed"
                    start_test_time = time.time()
                    self.current_mode = "fixed"

                    # Set fixed threshold (use original threshold regardless of distance)
                    self.hand_tracker.gesture_threshold = original_threshold
                else:
                    # ESC pressed during countdown
                    break

            elif test_phase == "testing_fixed":
                # Check time limit for this test (10 seconds)
                elapsed_test_time = time.time() - start_test_time
                remaining_time = 10 - elapsed_test_time

                if remaining_time <= 0:
                    # Switch back to distance selection
                    test_phase = "distance_selection"
                    # Display a completion message for a moment
                    completion_start = time.time()
                    while time.time() - completion_start < 2:  # Show for 2 seconds
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        processed_frame = self.hand_tracker.process_frame(
                            frame)
                        self.display_status(
                            processed_frame, "Pengujian dengan threshold tetap selesai!",
                            position=(30, 30), color=(0, 255, 0))
                        self.display_status(
                            processed_frame, "Kembali ke pemilihan jarak...",
                            position=(30, 70), color=(0, 255, 0))
                        cv2.imshow("Threshold Adaptation Test",
                                   processed_frame)
                        if cv2.waitKey(1) == 27:
                            break
                    continue

                # Display current test information
                self.display_status(
                    processed_frame, f"Mode: Threshold Tetap - Jarak: {self.current_distance}", position=(30, 30))
                self.display_status(
                    processed_frame, f"Waktu tersisa: {int(remaining_time)}s", position=(30, 70))
                self.display_status(
                    processed_frame, "Lakukan gerakan pointing dan selecting", position=(30, 110))
                self.display_status(
                    processed_frame, f"Nilai threshold saat ini: {self.hand_tracker.gesture_threshold:.2f}", position=(30, 150))

                # Check for gestures
                pointing_detected = self.hand_tracker.is_pointing()
                selecting_detected = self.hand_tracker.is_selecting()

                # Get the detected gesture name
                detected_gesture = None
                if pointing_detected:
                    detected_gesture = "pointing"
                elif selecting_detected:
                    detected_gesture = "selecting"

                # Record results for every frame
                results.append([
                    elapsed_test_time,          # Time since test started
                    frame_count,                # Frame number
                    self.current_mode,          # Adaptive or fixed threshold
                    self.current_distance,      # Current distance
                    self.hand_tracker.gesture_threshold,  # Current threshold value
                    detected_gesture,           # Detected gesture (or None)
                    1 if detected_gesture else 0  # 1 if detected, 0 if not
                ])

                # Display detection status
                if detected_gesture:
                    self.display_status(processed_frame, f"Terdeteksi: {detected_gesture}", position=(
                        30, 190), color=(0, 255, 0))
                else:
                    self.display_status(processed_frame, "Terdeteksi: Tidak", position=(
                        30, 190), color=(0, 0, 255))

                # Allow ESC to exit during testing
                if cv2.waitKey(1) == 27:  # ESC
                    break

            # Show frame
            cv2.imshow("Threshold Adaptation Test", processed_frame)

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        # Restore original threshold
        self.hand_tracker.gesture_threshold = original_threshold

        # Save results
        headers = ["timestamp", "frame_number", "mode",
                   "distance", "threshold", "gesture", "detected"]
        self.save_results(results, headers)

        print(
            f"Pengujian adaptasi threshold selesai. {len(results)} sampel data dicatat.")
