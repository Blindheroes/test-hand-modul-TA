"""
Lighting Test for Hand Gesture Recognition
Tests hand gesture recognition accuracy under different lighting conditions
"""
import cv2
import time
import numpy as np
from .base_test import BaseTest


class LightingTest(BaseTest):
    def __init__(self, camera=0, duration=20, results_dir="results"):
        super().__init__(camera, duration, results_dir)
        self.test_name = "lighting"
        # ~50 lux, ~300 lux, ~600 lux
        self.lighting_conditions = ["low", "medium", "high"]
        self.gestures = ["pointing", "selecting"]
        self.current_gesture_index = 0

    def countdown(self, cap, message):
        """Display a 5-second countdown with a specific message"""
        print(f"{message} - Test will begin in 5 seconds...")

        start_time = time.time()
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)  # Add small delay to prevent CPU overload
                continue

            # Calculate remaining time
            # Process frame with hand tracker (just for display)
            remaining = 5 - int(time.time() - start_time)
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
            cv2.imshow("Lighting Test", processed_frame)

            # Break loop if ESC key is pressed
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                return False

        print("Mulai pengujian!")
        return True

    def run(self):
        """Run the lighting test"""
        # Open camera
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera}")
            return

        results = []
        overall_start_time = time.time()
        test_phase = "setup"  # setup, gesture_selection, countdown, testing
        current_lighting = None
        current_gesture = None
        start_test_time = None
        frame_count = 0

        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                time.sleep(0.1)  # Add small delay to prevent CPU overload
                continue            # Process frame with hand tracker
            processed_frame = self.hand_tracker.process_frame(frame)
            # Add finger states visualization (skeleton and finger status)
            processed_frame = self.hand_tracker.visualize_finger_states(
                processed_frame)
            frame_count += 1

            # Estimate current frame brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)

            if test_phase == "setup":
                # Display instructions to select lighting condition
                self.display_status(
                    processed_frame, "Pilih kondisi pencahayaan:", position=(30, 30))
                self.display_status(
                    processed_frame, "1 = Rendah (~50 lux)", position=(30, 70))
                self.display_status(
                    processed_frame, "2 = Sedang (~300 lux)", position=(30, 110))
                self.display_status(
                    processed_frame, "3 = Tinggi (~600 lux)", position=(30, 150))
                self.display_status(
                    processed_frame, f"Kecerahan frame saat ini: {brightness:.1f}", position=(30, 190))
                self.display_status(
                    processed_frame, "ESC = Keluar", position=(30, 230))

                # Get user input for lighting
                key = cv2.waitKey(1)
                if key == ord('1'):
                    current_lighting = "low"
                    test_phase = "gesture_selection"
                elif key == ord('2'):
                    current_lighting = "medium"
                    test_phase = "gesture_selection"
                elif key == ord('3'):
                    current_lighting = "high"
                    test_phase = "gesture_selection"
                elif key == 27:  # ESC
                    break

            elif test_phase == "gesture_selection":
                # Display instructions to select gesture
                self.display_status(
                    processed_frame, f"Pencahayaan: {current_lighting}", position=(30, 30))
                self.display_status(
                    processed_frame, f"Kecerahan frame saat ini: {brightness:.1f}", position=(30, 70))
                self.display_status(
                    processed_frame, "Pilih gerakan pengujian:", position=(30, 110))
                self.display_status(
                    processed_frame, "1 = Pointing (jari telunjuk)", position=(30, 150))
                self.display_status(
                    processed_frame, "2 = Selecting (telunjuk dan kelingking)", position=(30, 190))
                self.display_status(
                    processed_frame, "0 = Kembali pilih pencahayaan", position=(30, 230))
                self.display_status(
                    processed_frame, "ESC = Keluar", position=(30, 270))

                # Get user input for gesture
                key = cv2.waitKey(1)
                if key == ord('1'):
                    current_gesture = "pointing"
                    test_phase = "countdown"
                elif key == ord('2'):
                    current_gesture = "selecting"
                    test_phase = "countdown"
                elif key == ord('0'):
                    current_lighting = None
                    test_phase = "setup"
                elif key == 27:  # ESC
                    break

            elif test_phase == "countdown":
                message = f"Bersiap untuk pengujian pencahayaan {current_lighting} dengan gerakan {current_gesture}"

                # Show countdown (we don't need to show the frame here, it's done in countdown method)
                if self.countdown(cap, message):
                    test_phase = "testing"
                    start_test_time = time.time()
                    frame_count = 0  # Reset frame counter for this test
                else:
                    # ESC pressed during countdown
                    break

            elif test_phase == "testing":
                # Check time limit for this test
                elapsed_test_time = time.time() - start_test_time
                remaining_time = self.duration - elapsed_test_time

                if remaining_time <= 0:
                    # Return to gesture selection for the same lighting condition
                    test_phase = "gesture_selection"
                    # Display a completion message for a moment
                    completion_start = time.time()
                    while time.time() - completion_start < 2:  # Show for 2 seconds
                        ret, frame = cap.read()
                        if not ret:
                            # Small delay to prevent CPU overload                            time.sleep(0.01)
                            continue
                        processed_frame = self.hand_tracker.process_frame(
                            frame)
                        # Add finger states visualization (skeleton and finger status)
                        processed_frame = self.hand_tracker.visualize_finger_states(
                            processed_frame)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        brightness = np.mean(gray)
                        self.display_status(
                            processed_frame, f"Pengujian {current_gesture} pada pencahayaan {current_lighting} selesai!",
                            position=(30, 30), color=(0, 255, 0))
                        self.display_status(
                            processed_frame, "Kembali ke menu pemilihan gerakan...",
                            position=(30, 70), color=(0, 255, 0))
                        cv2.imshow("Lighting Test", processed_frame)
                        if cv2.waitKey(1) == 27:
                            break
                    continue

                # Display current test information
                self.display_status(
                    processed_frame, f"Pencahayaan: {current_lighting}", position=(30, 30))
                self.display_status(
                    processed_frame, f"Gerakan: {current_gesture}", position=(30, 70))
                self.display_status(
                    processed_frame, f"Waktu tersisa: {int(remaining_time)}s", position=(30, 110))
                self.display_status(
                    processed_frame, f"Kecerahan frame: {brightness:.1f}", position=(30, 150))

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
                    current_lighting,        # Current lighting condition
                    current_gesture,         # Target gesture
                    brightness,              # Frame brightness
                    1 if is_detected else 0  # 1 if detected, 0 if not
                ])

                # Display detection status
                if is_detected:
                    self.display_status(processed_frame, "Terdeteksi: Ya ✓", position=(
                        30, 190), color=(0, 255, 0))
                else:
                    self.display_status(processed_frame, "Terdeteksi: Tidak ✗", position=(
                        30, 190), color=(0, 0, 255))

                # Allow ESC to exit during testing
                if cv2.waitKey(1) == 27:  # ESC
                    break

            # Show frame
            cv2.imshow("Lighting Test", processed_frame)

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        # Avoid problems if no results collected
        if not results:
            print("Pengujian dibatalkan atau tidak ada data yang dikumpulkan.")
            return

        # Save results
        headers = ["timestamp", "frame_number", "lighting",
                   "gesture", "brightness", "detected"]
        self.save_results(results, headers)

        print(
            f"Pengujian pencahayaan selesai. {len(results)} sampel data dicatat.")
