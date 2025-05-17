"""
Hand Depth Calculation Utilities for Hand Tracking
This module provides functions for calculating and visualizing hand depth.
"""

import cv2
import numpy as np
from collections import deque


def calculate_hand_depth(landmarks):
    """
    Calculate relative hand depth from camera using 3D Euclidean distance between key landmarks.

    This method uses multiple landmark pairs to get a more accurate estimation of how far 
    the hand is from the camera. It calculates distances between:
    1. Wrist to middle finger MCP
    2. Index finger MCP to pinky MCP
    3. Thumb CMC to pinky MCP

    Args:
        landmarks: MediaPipe hand landmarks for a single hand

    Returns:
        Relative depth value (higher value means hand is closer to camera)
        or None if no hand is detected
    """
    if not landmarks:
        return None

    # Key points for distance calculation
    wrist = landmarks[0]  # Wrist landmark
    index_mcp = landmarks[5]  # Index finger MCP joint
    middle_mcp = landmarks[9]  # Middle finger MCP joint
    pinky_mcp = landmarks[17]  # Pinky finger MCP joint
    thumb_cmc = landmarks[1]  # Thumb CMC joint

    # Calculate Euclidean 3D distances between landmark pairs
    wrist_to_middle = np.sqrt(
        (wrist.x - middle_mcp.x)**2 +
        (wrist.y - middle_mcp.y)**2 +
        (wrist.z - middle_mcp.z)**2
    )

    index_to_pinky = np.sqrt(
        (index_mcp.x - pinky_mcp.x)**2 +
        (index_mcp.y - pinky_mcp.y)**2 +
        (index_mcp.z - pinky_mcp.z)**2
    )

    thumb_to_pinky = np.sqrt(
        (thumb_cmc.x - pinky_mcp.x)**2 +
        (thumb_cmc.y - pinky_mcp.y)**2 +
        (thumb_cmc.z - pinky_mcp.z)**2
    )

    # Combine the measurements (weighted average)
    # Weights prioritize the most reliable measurements
    depth = (0.4 * wrist_to_middle + 0.4 *
             index_to_pinky + 0.2 * thumb_to_pinky)

    return depth


class DepthTracker:
    """Class for tracking and smoothing hand depth values over time"""

    def __init__(self):
        self.depth_history = deque(maxlen=10)  # For smoothing depth values
        self.depth_min = float('inf')  # Min depth value observed
        self.depth_max = 0  # Max depth value observed
        self.current_depth = 0

    def update(self, landmarks):
        """
        Update depth tracking with new landmarks

        Args:
            landmarks: MediaPipe hand landmarks from current frame

        Returns:
            Current smoothed depth value
        """
        if not landmarks:
            return self.current_depth

        # Calculate new depth value
        depth = calculate_hand_depth(landmarks[0].landmark)

        # Update min/max for normalization
        if depth < self.depth_min:
            self.depth_min = depth
        if depth > self.depth_max:
            self.depth_max = max(depth, self.depth_max)

        # Add to history for smoothing
        self.depth_history.append(depth)

        # Calculate smoothed depth using moving average
        if len(self.depth_history) > 0:
            self.current_depth = sum(
                self.depth_history) / len(self.depth_history)
        else:
            self.current_depth = depth

        return self.current_depth

    def get_normalized_depth(self):
        """
        Get depth normalized to 0-100% scale

        Returns:
            Normalized depth as percentage (0-100)
        """
        depth_range = max(0.001, self.depth_max -
                          self.depth_min)  # Prevent division by zero
        return min(100, max(0, 100 * (self.current_depth - self.depth_min) / depth_range))


def visualize_hand_depth(frame, depth_tracker):
    """
    Add a visual indicator of hand depth to the frame

    Args:
        frame: Image frame to draw on
        depth_tracker: DepthTracker instance with depth information

    Returns:
        Frame with depth visualization
    """
    if depth_tracker.current_depth == 0:
        return frame

    # Get normalized depth (0-100%)
    normalized_depth = depth_tracker.get_normalized_depth()

    # Depth category
    if normalized_depth < 30:
        depth_text = "Jauh"
        depth_color = (0, 0, 255)  # Red (BGR)
    elif normalized_depth < 70:
        depth_text = "Sedang"
        depth_color = (0, 255, 255)  # Yellow (BGR)
    else:
        depth_text = "Dekat"
        depth_color = (0, 255, 0)  # Green (BGR)

    # Display depth text and value
    cv2.putText(
        frame,
        f"Jarak Tangan: {depth_text} ({normalized_depth:.1f}%)",
        (10, frame.shape[0] - 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        depth_color,
        2
    )

    # Draw depth bar
    bar_width = 150
    bar_height = 20
    bar_x = 10
    bar_y = frame.shape[0] - 100

    # Draw bar background
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (50, 50, 50),
        -1
    )

    # Draw filled portion of bar
    filled_width = int((normalized_depth / 100) * bar_width)
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + filled_width, bar_y + bar_height),
        depth_color,
        -1
    )

    # Draw border around bar
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (200, 200, 200),
        1
    )

    return frame
