"""
Base Test Class for Virtual Try-On Game testing
"""
import os
import time
import datetime
import cv2
import csv
import numpy as np
from hand_tracking import HandTracker


class BaseTest:
    def __init__(self, camera=0, duration=60, results_dir="results"):
        """
        Initialize the base test

        Args:
            camera: Camera index to use
            duration: Test duration in seconds
            results_dir: Directory to save results
        """
        self.camera = camera
        self.duration = duration
        self.results_dir = results_dir
        self.hand_tracker = HandTracker()
        self.test_name = "base"  # Override in subclasses

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

    def get_timestamp(self):
        """Generate a timestamp string for file naming"""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_results_filename(self):
        """Get the filename for results CSV"""
        return os.path.join(self.results_dir, f"{self.test_name}_test_{self.get_timestamp()}.csv")

    def run(self):
        """Run the test - override in subclasses"""
        raise NotImplementedError("Subclasses must implement run() method")

    def save_results(self, results, headers):
        """
        Save results to a CSV file

        Args:
            results: List of result rows
            headers: List of column headers
        """
        filename = self.get_results_filename()

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(results)

        print(f"Results saved to {filename}")

    def display_status(self, frame, text, position=(30, 30), color=(0, 0, 255)):
        """
        Display status text on frame

        Args:
            frame: Frame to display text on
            text: Text to display
            position: Position tuple (x, y)
            color: Color tuple (B, G, R)
        """
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
