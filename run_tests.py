#!/usr/bin/env python
"""
Test Runner for Virtual Try-On Game
This script runs various tests for the hand tracking module.
"""
import argparse
import os
import time
import datetime
import cv2
import mediapipe as mp
import numpy as np
import csv
from hand_tracking import HandTracker
import importlib

# Import test modules
from tests.distance_test import DistanceTest
from tests.lighting_test import LightingTest
from tests.threshold_test import ThresholdTest
from tests.temporal_test import TemporalTest


def get_timestamp():
    """Generate a timestamp string for file naming"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def create_results_dir():
    """Ensure the results directory exists"""
    os.makedirs("results", exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for the Virtual Try-On Game")
    parser.add_argument("--test", type=str, required=True,
                        choices=["all", "distance", "lighting",
                                 "threshold", "temporal"],
                        help="Test to run (all, distance, lighting, threshold, temporal)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index to use (default: 0)")

    args = parser.parse_args()
    create_results_dir()

    # Run selected test(s)
    if args.test == "all" or args.test == "distance":
        print("\nMulai pengujian jarak...")
        distance_test = DistanceTest(camera=args.camera)
        distance_test.run()

    if args.test == "all" or args.test == "lighting":
        print("\nMulai pengujian pencahayaan...")
        lighting_test = LightingTest(camera=args.camera)
        lighting_test.run()

    if args.test == "all" or args.test == "threshold":
        print("\nMulai pengujian adaptasi threshold...")
        threshold_test = ThresholdTest(camera=args.camera)
        threshold_test.run()

    if args.test == "all" or args.test == "temporal":
        print("\nMulai pengujian pemfilteran temporal...")
        temporal_test = TemporalTest(camera=args.camera)
        temporal_test.run()

    print("\nSemua pengujian selesai. Hasil disimpan di folder 'results'.")


if __name__ == "__main__":
    main()
