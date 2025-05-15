"""
Temporal Filtering Test for Hand Gesture Recognition
Tests the effectiveness of temporal filtering for hand gesture recognition
"""
import cv2
import time
import numpy as np
from collections import deque
from .base_test import BaseTest


class TemporalTest(BaseTest):
    def __init__(self, camera=0, duration=20, results_dir="results"):
        super().__init__(camera, duration, results_dir)
        self.test_name = "temporal"
        self.filtering_modes = ["with_filtering", "without_filtering"]
        self.current_mode = None  # Will be selected during test

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
            # Add finger states visualization (skeleton and finger status)
            processed_frame = self.hand_tracker.visualize_finger_states(
                processed_frame)

            # Display countdown on frame
            self.display_status(processed_frame, message,
                                position=(50, 50), color=(0, 0, 255))
            self.display_status(processed_frame, f"Mulai dalam: {remaining} detik", position=(
                50, 100), color=(0, 0, 255))

            # Show the frame
            cv2.imshow("Temporal Filtering Test", processed_frame)

            # Break loop if ESC key is pressed
            if cv2.waitKey(1) == 27:  # ESC key
                return False

        print("Mulai pengujian!")
        return True

    def run(self):
        """Run the temporal filtering test"""
        # Open camera
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera}")
            return

        results = []
        overall_start_time = time.time()
        # mode_selection, countdown_with_filtering, testing_with_filtering, countdown_without_filtering, testing_without_filtering
        test_phase = "mode_selection"
        start_test_time = None
        frame_count = 0

        # Store original gesture history maxlen
        original_history_maxlen = self.hand_tracker.gesture_history['pointing'].maxlen

        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Process frame with hand tracker
            processed_frame = self.hand_tracker.process_frame(frame)
            # Add finger states visualization (skeleton and finger status)
            processed_frame = self.hand_tracker.visualize_finger_states(
                processed_frame)
            frame_count += 1

            if test_phase == "mode_selection":
                # Display instructions
                self.display_status(
                    processed_frame, "Pengujian Pemfilteran Temporal", position=(30, 30))
                self.display_status(
                    processed_frame, "Pilih mode pemfilteran:", position=(30, 70))
                self.display_status(
                    processed_frame, "1 = Mode Dengan Pemfilteran", position=(30, 110))
                self.display_status(
                    processed_frame, "2 = Mode Tanpa Pemfilteran", position=(30, 150))
                self.display_status(
                    processed_frame, "ESC = Keluar", position=(30, 190))

                # Get user input for mode selection
                key = cv2.waitKey(1)
                if key == ord("1"):
                    self.current_mode = "with_filtering"
                    test_phase = "countdown_with_filtering"
                elif key == ord("2"):
                    self.current_mode = "without_filtering"
                    test_phase = "countdown_without_filtering"
                elif key == 27:  # ESC
                    break

            elif test_phase == "countdown_with_filtering":
                message = "Bersiap untuk pengujian dengan pemfilteran temporal"

                # Show countdown
                cv2.imshow("Temporal Filtering Test", processed_frame)
                if self.countdown(cap, message):
                    test_phase = "testing_with_filtering"
                    start_test_time = time.time()

                    # Enable temporal filtering
                    self.hand_tracker.toggle_features(
                        use_temporal_filtering=True)
                    for gesture in self.hand_tracker.gesture_history:
                        self.hand_tracker.gesture_history[gesture] = deque(
                            maxlen=original_history_maxlen)
                else:
                    # ESC pressed during countdown
                    break

            elif test_phase == "testing_with_filtering":
                # Check time limit for this test (20 seconds)
                elapsed_test_time = time.time() - start_test_time
                remaining_time = self.duration - elapsed_test_time

                if remaining_time <= 0:
                    # Switch back to mode selection
                    test_phase = "mode_selection"
                    # Display a completion message for a moment
                    completion_start = time.time()
                    while time.time() - completion_start < 2:  # Show for 2 seconds
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        processed_frame = self.hand_tracker.process_frame(
                            frame)
                        self.display_status(
                            processed_frame, "Pengujian dengan pemfilteran selesai!",
                            position=(30, 30), color=(0, 255, 0))
                        cv2.imshow("Temporal Filtering Test", processed_frame)
                        if cv2.waitKey(1) == 27:
                            break
                    continue

                # Display current test information
                self.display_status(
                    processed_frame, f"Mode: Dengan Pemfilteran", position=(30, 30))
                self.display_status(
                    processed_frame, f"Waktu tersisa: {int(remaining_time)}s", position=(30, 70))
                self.display_status(
                    processed_frame, "Lakukan gerakan pointing dan selecting dengan cepat bergantian", position=(30, 110))

                # Check for gestures (raw detection - before filtering)
                pointing_raw = bool(
                    self.hand_tracker.gesture_history['pointing'][-1]) if self.hand_tracker.gesture_history['pointing'] else False
                selecting_raw = bool(
                    self.hand_tracker.gesture_history['selecting'][-1]) if self.hand_tracker.gesture_history['selecting'] else False

                # Filtered detection (after temporal filtering)
                pointing_filtered = self.hand_tracker.is_pointing()
                selecting_filtered = self.hand_tracker.is_selecting()

                # Get the detected raw gesture name
                detected_raw_gesture = None
                if pointing_raw and selecting_raw:
                    # If both are detected raw, prioritize the strongest signal
                    pointing_confidence = self.hand_tracker._get_gesture_confidence(
                        'pointing')
                    selecting_confidence = self.hand_tracker._get_gesture_confidence(
                        'selecting')
                    if selecting_confidence > pointing_confidence:
                        detected_raw_gesture = "selecting"
                    else:
                        detected_raw_gesture = "pointing"
                elif pointing_raw:
                    detected_raw_gesture = "pointing"
                elif selecting_raw:
                    detected_raw_gesture = "selecting"

                # Get the detected filtered gesture name
                detected_filtered_gesture = None
                if pointing_filtered and selecting_filtered:
                    # If both are detected after filtering, prioritize the strongest signal
                    pointing_confidence = self.hand_tracker._get_gesture_confidence(
                        'pointing')
                    selecting_confidence = self.hand_tracker._get_gesture_confidence(
                        'selecting')
                    if selecting_confidence > pointing_confidence:
                        detected_filtered_gesture = "selecting"
                    else:
                        detected_filtered_gesture = "pointing"
                elif pointing_filtered:
                    detected_filtered_gesture = "pointing"
                elif selecting_filtered:
                    detected_filtered_gesture = "selecting"

                # Record results for every frame
                results.append([
                    elapsed_test_time,                      # Time since test started
                    self.current_mode,                      # With or without filtering
                    # Use filtered gesture, fallback to raw if filtered is None
                    detected_filtered_gesture or detected_raw_gesture,
                    "filtered" if detected_filtered_gesture else "raw"  # Detection type
                ])

                # Display detection status
                if detected_raw_gesture:
                    self.display_status(processed_frame, f"Deteksi Raw: {detected_raw_gesture}",
                                        position=(30, 150), color=(255, 0, 0))
                else:
                    self.display_status(processed_frame, "Deteksi Raw: Tidak",
                                        position=(30, 150), color=(255, 0, 0))

                if detected_filtered_gesture:
                    self.display_status(processed_frame, f"Deteksi Terfilter: {detected_filtered_gesture}",
                                        position=(30, 190), color=(0, 255, 0))
                else:
                    self.display_status(processed_frame, "Deteksi Terfilter: Tidak",
                                        position=(30, 190), color=(0, 255, 0))

                # Allow ESC to exit during testing
                if cv2.waitKey(1) == 27:  # ESC
                    break

            elif test_phase == "countdown_without_filtering":
                message = "Bersiap untuk pengujian tanpa pemfilteran temporal"

                # Show countdown
                cv2.imshow("Temporal Filtering Test", processed_frame)
                if self.countdown(cap, message):
                    test_phase = "testing_without_filtering"
                    start_test_time = time.time()

                    # Disable temporal filtering by setting history length to 1
                    self.hand_tracker.toggle_features(
                        use_temporal_filtering=False)
                    for gesture in self.hand_tracker.gesture_history:
                        self.hand_tracker.gesture_history[gesture] = deque(
                            maxlen=1)
                else:
                    # ESC pressed during countdown
                    break

            elif test_phase == "testing_without_filtering":
                # Check time limit for this test (20 seconds)
                elapsed_test_time = time.time() - start_test_time
                remaining_time = self.duration - elapsed_test_time

                if remaining_time <= 0:
                    # Switch back to mode selection
                    test_phase = "mode_selection"
                    # Display a completion message for a moment
                    completion_start = time.time()
                    while time.time() - completion_start < 2:  # Show for 2 seconds
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        processed_frame = self.hand_tracker.process_frame(
                            frame)
                        self.display_status(
                            processed_frame, "Pengujian tanpa pemfilteran selesai!",
                            position=(30, 30), color=(0, 255, 0))
                        cv2.imshow("Temporal Filtering Test", processed_frame)
                        if cv2.waitKey(1) == 27:
                            break
                    continue

                # Display current test information
                self.display_status(
                    processed_frame, f"Mode: Tanpa Pemfilteran", position=(30, 30))
                self.display_status(
                    processed_frame, f"Waktu tersisa: {int(remaining_time)}s", position=(30, 70))
                self.display_status(
                    processed_frame, "Lakukan gerakan pointing dan selecting dengan cepat bergantian", position=(30, 110))

                # Check for raw gestures - these will be the same as filtered without temporal filtering
                # since history length is 1
                pointing_raw = bool(
                    self.hand_tracker.gesture_history['pointing'][-1]) if self.hand_tracker.gesture_history['pointing'] else False
                selecting_raw = bool(
                    self.hand_tracker.gesture_history['selecting'][-1]) if self.hand_tracker.gesture_history['selecting'] else False

                # Get the detected raw gesture name
                detected_raw_gesture = None
                if pointing_raw and selecting_raw:
                    # If both are detected raw, prioritize the strongest signal
                    pointing_confidence = self.hand_tracker._get_gesture_confidence(
                        'pointing')
                    selecting_confidence = self.hand_tracker._get_gesture_confidence(
                        'selecting')
                    if selecting_confidence > pointing_confidence:
                        detected_raw_gesture = "selecting"
                    else:
                        detected_raw_gesture = "pointing"
                elif pointing_raw:
                    detected_raw_gesture = "pointing"
                elif selecting_raw:
                    detected_raw_gesture = "selecting"

                # Record results for every frame
                results.append([
                    elapsed_test_time,          # Time since test started
                    self.current_mode,          # With or without filtering
                    detected_raw_gesture,       # Raw gesture detection
                    # Detection type (always raw without filtering)
                    "raw"
                ])

                # Display detection status
                if detected_raw_gesture:
                    self.display_status(processed_frame, f"Deteksi Raw: {detected_raw_gesture}",
                                        position=(30, 150), color=(255, 0, 0))
                else:
                    self.display_status(processed_frame, "Deteksi Raw: Tidak",
                                        position=(30, 150), color=(255, 0, 0))

                # For consistency with filtered mode UI, also show filtered detection
                # (which should be same as raw in this mode)
                self.display_status(processed_frame,
                                    f"Deteksi Terfilter: {detected_raw_gesture}" if detected_raw_gesture else "Deteksi Terfilter: Tidak",
                                    position=(30, 190), color=(0, 255, 0))

                # Allow ESC to exit during testing
                if cv2.waitKey(1) == 27:  # ESC
                    break

            # Show frame
            cv2.imshow("Temporal Filtering Test", processed_frame)

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        # Restore original history length
        self.hand_tracker.toggle_features(use_temporal_filtering=True)
        for gesture in self.hand_tracker.gesture_history:
            self.hand_tracker.gesture_history[gesture] = deque(
                maxlen=original_history_maxlen)

        # Save results
        headers = ["timestamp", "mode", "gesture", "detection_type"]
        self.save_results(results, headers)

        print(
            f"Pengujian pemfilteran temporal selesai. {len(results)} sampel data dicatat.")
