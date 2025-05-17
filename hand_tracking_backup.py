"""
Hand Tracking Module for Virtual Try-On Game
This module handles hand gesture recognition for controlling the virtual try-on interface.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque


class HandTracker:
    def __init__(self, use_threshold_adaptation=True, use_temporal_filtering=True):
        # Initialize MediaPipe hands with optimized parameters
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,   # Set to False for video processing
            max_num_hands=1,           # Track only one hand for simplicity
            min_detection_confidence=0.6,  # Increased from 0.5 for higher confidence
            min_tracking_confidence=0.6,   # Increased from 0.5 for more stable tracking
            model_complexity=1         # 1 provides a good balance between speed and accuracy
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Finger landmarks
        self.landmarks = None
        self.frame_height = 0
        self.frame_width = 0        # Flags for enabling/disabling features (for testing)
        self.use_threshold_adaptation = use_threshold_adaptation
        self.use_temporal_filtering = use_temporal_filtering
        
        # Hand distance estimation
        # Relative z-coordinate of the hand (depth)
        self.hand_z = 0
        # Enhanced depth measurement using multiple landmark pairs
        self.hand_depth = 0
        self.depth_history = deque(maxlen=10)  # Smoothing filter for depth values
        self.depth_reference = None  # Reference value for calibrating depth
        self.depth_min = float('inf')  # Minimum depth value observed
        self.depth_max = 0  # Maximum depth value observed
        
        # Finger angle thresholds (in degrees)
        self.extension_threshold = 160  # Angle above which a finger is considered extended
        # Angle below which a finger is considered flexed/closed
        self.flexion_threshold = 120
        # General gesture detection threshold for adaptability testing
        self.gesture_threshold = 0.5
        # Gesture history for temporal filtering
        self.gesture_history = {
            'pointing': deque(maxlen=5),
            'selecting': deque(maxlen=5),
            'grabbing': deque(maxlen=5),
            'open_palm': deque(maxlen=5)
        }

        # Last detection time for tracking stability
        self.last_detection_time = time.time()

    def process_frame(self, frame):
        """
        Process the frame and detect hand landmarks

        Args:
            frame: RGB image frame

        Returns:
            Frame with hand landmarks drawn (if enabled)
        """
        self.frame_height, self.frame_width = frame.shape[:2]

        # Convert to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hands
        results = self.hands.process(rgb_frame)

        current_time = time.time()        # Update landmarks if hands are detected
        if results.multi_hand_landmarks:
            self.landmarks = results.multi_hand_landmarks
            self.last_detection_time = current_time

            # Calculate enhanced hand depth using multiple landmark pairs
            self.calculate_hand_depth()
            
            # Also update the basic depth measure for backward compatibility
            wrist = self.landmarks[0].landmark[0]
            middle_mcp = self.landmarks[0].landmark[9]
            
            # Calculate 3D distance between these points
            dist = np.sqrt(
                (wrist.x - middle_mcp.x)**2 +
                (wrist.y - middle_mcp.y)**2 +
                (wrist.z - middle_mcp.z)**2
            )

            # Update hand z-coordinate (depth)
            self.hand_z = dist
        elif current_time - self.last_detection_time > 0.5:
            # Clear landmarks if no detection for 0.5 seconds
            self.landmarks = None

        return frame

    def calculate_hand_depth(self):
        """
        Calculate relative hand depth from camera using 3D Euclidean distance between key landmarks.
        
        This method uses multiple landmark pairs to get a more accurate estimation of how far 
        the hand is from the camera. It calculates distances between:
        1. Wrist to middle finger MCP
        2. Index finger MCP to pinky MCP
        3. Thumb CMC to pinky MCP
        
        Returns:
            Relative depth value (higher value means hand is closer to camera)
            or None if no hand is detected
        """
        if not self.landmarks:
            return None
            
        landmarks = self.landmarks[0].landmark
        
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
        depth = (0.4 * wrist_to_middle + 0.4 * index_to_pinky + 0.2 * thumb_to_pinky)
        
        # Keep track of min and max depth values for normalization
        if depth < self.depth_min:
            self.depth_min = depth
        if depth > self.depth_max:
            self.depth_max = max(depth, self.depth_max)
            
        # Add to history for smoothing
        self.depth_history.append(depth)
        
        # Calculate smoothed depth using a moving average
        if len(self.depth_history) > 0:
            self.hand_depth = sum(self.depth_history) / len(self.depth_history)
        else:
            self.hand_depth = depth
            
        return self.hand_depth

    def get_pointer_position(self):
        """
        Get the position of the index finger tip (pointer)

        Returns:
            (x, y) tuple of the pointer position, or None if not detected
        """
        if not self.landmarks:
            return None

        # Get the index finger tip position (landmark 8)
        hand_landmarks = self.landmarks[0]  # Use first hand only
        index_tip = hand_landmarks.landmark[8]

        # Convert normalized coordinates to pixel coordinates

        x = int(index_tip.x * self.frame_width)
        y = int(index_tip.y * self.frame_height)

        return (x, y)

    def _calculate_angle(self, p1, p2, p3):
        """
        Calculate the angle between three points

        Args:
            p1, p2, p3: Three points where p2 is the middle point (joint)

        Returns:
            Angle in degrees
        """
        # Convert landmarks to numpy arrays
        p1_array = np.array([p1.x, p1.y, p1.z])
        p2_array = np.array([p2.x, p2.y, p2.z])
        p3_array = np.array([p3.x, p3.y, p3.z])

        # Calculate vectors
        v1 = p1_array - p2_array
        v2 = p3_array - p2_array

        # Calculate angle using dot product
        cosine_angle = np.dot(v1, v2) / \
            (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        # Convert to degrees
        return math.degrees(angle)

    def get_finger_angles(self):
        """
        Calculate the angles for each finger

        Returns:
            Dictionary of finger angles or None if no hand detected
        """
        if not self.landmarks:
            return None

        hand_landmarks = self.landmarks[0]
        landmarks = hand_landmarks.landmark

        # Landmark indices for each finger joint
        # Format: [finger_name, [mcp, pip, dip, tip]]
        finger_indices = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

        angles = {}

        # Calculate angle for each finger
        for finger_name, indices in finger_indices.items():
            # For non-thumb fingers
            if finger_name != 'thumb':
                # Angle at MCP joint (between palm and proximal phalanx)
                mcp_angle = self._calculate_angle(
                    landmarks[0],      # wrist
                    landmarks[indices[0]],  # mcp
                    landmarks[indices[1]]   # pip
                )

                # Angle at PIP joint (between proximal and middle phalanx)
                pip_angle = self._calculate_angle(
                    landmarks[indices[0]],  # mcp
                    landmarks[indices[1]],  # pip
                    landmarks[indices[2]]   # dip
                )

                angles[finger_name] = {
                    'mcp': mcp_angle,
                    'pip': pip_angle
                }
            else:
                # Special case for thumb
                cmc_angle = self._calculate_angle(
                    landmarks[0],      # wrist
                    landmarks[indices[0]],  # cmc
                    landmarks[indices[1]]   # mcp
                )

                mcp_angle = self._calculate_angle(
                    landmarks[indices[0]],  # cmc
                    landmarks[indices[1]],  # mcp
                    landmarks[indices[2]]   # ip
                )

                angles[finger_name] = {
                    'cmc': cmc_angle,
                    'mcp': mcp_angle
                }

        return angles

    def is_finger_extended(self, finger_name, threshold_modifier=1.0):
        """
        Determine if a specific finger is extended based on joint angles

        Args:
            finger_name: String name of the finger ('thumb', 'index', 'middle', 'ring', 'pinky')
            threshold_modifier: Modifier to adjust the extension threshold dynamically

        Returns:
            Boolean indicating if the finger is extended
        """
        angles = self.get_finger_angles()
        if not angles:
            return False

        # Apply threshold_modifier only if threshold adaptation is enabled
        if self.use_threshold_adaptation:
            adjusted_threshold = self.extension_threshold * threshold_modifier
        else:
            adjusted_threshold = self.extension_threshold

        if finger_name == 'thumb':
            # Thumb is extended if both CMC and MCP joints are relatively straight
            # Slightly lower thresholds for more lenient detection
            return angles[finger_name]['cmc'] > adjusted_threshold * 0.65 and \
                angles[finger_name]['mcp'] > adjusted_threshold * 0.75
        else:
            # For other fingers, check MCP and PIP joints
            # A finger is extended if MCP and PIP joints are relatively straight
            # More relaxed condition for index finger to improve pointing detection
            if finger_name == 'index':
                return angles[finger_name]['mcp'] > adjusted_threshold * 0.9 and \
                    angles[finger_name]['pip'] > adjusted_threshold * 0.75
            else:
                return angles[finger_name]['mcp'] > adjusted_threshold and \
                    angles[finger_name]['pip'] > adjusted_threshold * 0.8

    def _update_gesture_history(self, gesture_name, detected):
        """
        Update gesture history for temporal filtering

        Args:
            gesture_name: Name of the gesture ('pointing', 'selecting', etc.)
            detected: Boolean indicating if the gesture was detected in current frame
      """
        if self.use_temporal_filtering:
            self.gesture_history[gesture_name].append(1 if detected else 0)
        else:
            # When temporal filtering is disabled, just store the latest value
            if self.gesture_history[gesture_name]:
                self.gesture_history[gesture_name].clear()
            self.gesture_history[gesture_name].append(1 if detected else 0)

    def _get_gesture_confidence(self, gesture_name):
        """
          Calculate confidence score for a gesture based on recent history

          Args:
              gesture_name: Name of the gesture to check

          Returns:
              Confidence score between 0.0 and 1.0, or raw value if temporal filtering is disabled
          """
        if not self.use_temporal_filtering:
            # Return 1.0 if detected in current frame, 0.0 otherwise
            return float(self.gesture_history[gesture_name][-1]) if self.gesture_history[gesture_name] else 0.0
        if not self.gesture_history[gesture_name]:
            return 0.0
        # Calculate weighted average (more recent detections have higher weight)
        weights = np.linspace(0.5, 1.0, len(
            self.gesture_history[gesture_name]))
        confidence = np.average(
            self.gesture_history[gesture_name],
            weights=weights
        )
        return confidence

    def is_pointing(self):
        """
        Determine if the hand is making a pointing gesture
        (only index finger extended, all others closed)

        Returns:
            Boolean indicating whether pointing gesture is detected
        """
        if not self.landmarks:
            self._update_gesture_history('pointing', False)
            return False

        # Adapt threshold based on hand distance (z-coordinate)
        # When hand is further away, be more lenient with angle requirements
        distance_factor = min(
            1.0, max(0.7, 1.0 - self.hand_z * 2)) if self.use_threshold_adaptation else 1.0
        # Check finger states using angle-based detection

        index_extended = self.is_finger_extended(
            'index', threshold_modifier=distance_factor)

        # Raw detection result - only check that index is extended and other fingers except thumb are closed
        # Thumb can be in any position (more lenient approach)
        # Update history
        detected = index_extended
        self._update_gesture_history('pointing', detected)

        # Return true if confidence exceeds threshold (using the adjustable gesture_threshold)
        return self._get_gesture_confidence('pointing') > self.gesture_threshold

    def is_selecting(self):
        """
        Determine if the selection gesture is being made
        (index finger and little finger extended, others closed)

        Returns:
            Boolean indicating whether selection gesture is detected
        """
        if not self.landmarks:
            self._update_gesture_history('selecting', False)
            return False

        # Adapt threshold based on hand distance (z-coordinate)
        distance_factor = min(
            1.0, max(0.7, 1.0 - self.hand_z * 2)) if self.use_threshold_adaptation else 1.0

        # Check finger states using angle-based detection with distance adaptation
        index_extended = self.is_finger_extended(
            'index', threshold_modifier=distance_factor)
        middle_extended = self.is_finger_extended(
            'middle', threshold_modifier=distance_factor)
        ring_extended = self.is_finger_extended(
            'ring', threshold_modifier=distance_factor)
        pinky_extended = self.is_finger_extended(
            'pinky', threshold_modifier=distance_factor)

        # Raw detection result
        # Update history
        detected = index_extended and pinky_extended 
        self._update_gesture_history('selecting', detected)

        # Return true if confidence exceeds threshold (using a slightly higher value than gesture_threshold)
        return self._get_gesture_confidence('selecting') > (self.gesture_threshold )

    def is_grabbing(self):
        """
        Determine if the hand is making a grabbing gesture
        (all fingers closed/flexed)

        Returns:
            Boolean indicating whether grabbing gesture is detected
        """
        if not self.landmarks:
            self._update_gesture_history('grabbing', False)
            return False

        # Adapt threshold based on hand distance (z-coordinate)
        distance_factor = min(
            1.0, max(0.7, 1.0 - self.hand_z * 2)) if self.use_threshold_adaptation else 1.0

        # Check if all fingers are flexed using distance-adapted thresholds
        thumb_extended = self.is_finger_extended(
            'thumb', threshold_modifier=distance_factor)
        index_extended = self.is_finger_extended(
            'index', threshold_modifier=distance_factor)
        middle_extended = self.is_finger_extended(
            'middle', threshold_modifier=distance_factor)
        ring_extended = self.is_finger_extended(
            'ring', threshold_modifier=distance_factor)
        pinky_extended = self.is_finger_extended(
            'pinky', threshold_modifier=distance_factor)

        # Raw detection result - no fingers should be extended
        detected = not any([thumb_extended, index_extended,
                           middle_extended, ring_extended, pinky_extended])

        # Update history
        self._update_gesture_history('grabbing', detected)

        # Return true if confidence exceeds threshold
        # Higher threshold for grabbing
        return self._get_gesture_confidence('grabbing') > 0.7

    def is_open_palm(self):
        """
        Determine if the hand is making an open palm gesture
        (all fingers extended)

        Returns:
            Boolean indicating whether open palm gesture is detected
        """
        if not self.landmarks:
            self._update_gesture_history('open_palm', False)
            return False

        # Adapt threshold based on hand distance (z-coordinate)
        distance_factor = min(
            1.0, max(0.7, 1.0 - self.hand_z * 2)) if self.use_threshold_adaptation else 1.0

        # Check if all fingers are extended using distance-adapted thresholds
        thumb_extended = self.is_finger_extended(
            'thumb', threshold_modifier=distance_factor)
        index_extended = self.is_finger_extended(
            'index', threshold_modifier=distance_factor)
        middle_extended = self.is_finger_extended(
            'middle', threshold_modifier=distance_factor)
        ring_extended = self.is_finger_extended(
            'ring', threshold_modifier=distance_factor)
        pinky_extended = self.is_finger_extended(
            'pinky', threshold_modifier=distance_factor)

        # Raw detection result - all fingers should be extended
        detected = all([thumb_extended, index_extended,
                       middle_extended, ring_extended, pinky_extended])

        # Update history
        self._update_gesture_history('open_palm', detected)

        # Return true if confidence exceeds threshold
        return self._get_gesture_confidence('open_palm') > 0.65

    def adjust_thresholds(self, extension_threshold=None, flexion_threshold=None):
        """
        Adjust the angle thresholds for finger state detection

        Args:
            extension_threshold: Angle above which a finger is considered extended (in degrees)
            flexion_threshold: Angle below which a finger is considered flexed/closed (in degrees)
        """        if extension_threshold is not None:
            self.extension_threshold = extension_threshold
        if flexion_threshold is not None:
            self.flexion_threshold = flexion_threshold
            
    def visualize_finger_states(self, frame):
        """
        Draw visual indicators for finger states on the frame

        Args:
            frame: Image frame to draw on

        Returns:
            Frame with finger state visualization
        """
        if not self.landmarks:
            return frame

        # Get finger states
        finger_states = {
            'thumb': self.is_finger_extended('thumb'),
            'index': self.is_finger_extended('index'),
            'middle': self.is_finger_extended('middle'),
            'ring': self.is_finger_extended('ring'),
            'pinky': self.is_finger_extended('pinky')
        }

        # Draw hand landmarks
        for hand_landmarks in self.landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        # Add finger state indicators
        y_pos = 30
        for finger, is_extended in finger_states.items():
            color = (0, 255, 0) if is_extended else (
                0, 0, 255)  # Green if extended, red if flexed
            cv2.putText(frame, f"{finger}: {'Extended' if is_extended else 'Flexed'}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30

        # Add current gesture label
        gesture = "None"
        if self.is_selecting():
            gesture = "Selecting"
        elif self.is_pointing():
            gesture = "Pointing"
        elif self.is_grabbing():
            gesture = "Grabbing"
        elif self.is_open_palm():
            gesture = "Open Palm"

        cv2.putText(frame, f"Gesture: {gesture}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_pos += 30

        # Display feature status (for testing purposes)
        adapt_status = "ON" if self.use_threshold_adaptation else "OFF"
        filter_status = "ON" if self.use_temporal_filtering else "OFF"
        cv2.putText(frame, f"Threshold Adaptation: {adapt_status}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 30
        cv2.putText(frame, f"Temporal Filtering: {filter_status}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add hand depth visualization
        frame = self.visualize_hand_depth(frame)
        
        return frame

    def visualize_hand_depth(self, frame):
        """
        Add a visual indicator of hand depth to the frame
        
        Args:
            frame: Image frame to draw on
            
        Returns:
            Frame with depth visualization
        """
        if not self.landmarks or self.hand_depth == 0:
            return frame
            
        # Normalize depth value for visualization (0-100%)
        # A higher value means hand is closer to camera
        depth_range = max(0.001, self.depth_max - self.depth_min)  # Prevent division by zero
        normalized_depth = min(100, max(0, 100 * (self.hand_depth - self.depth_min) / depth_range))
        
        # Depth category
        if normalized_depth < 30:
            depth_text = "Far"
            depth_color = (0, 0, 255)  # Red
        elif normalized_depth < 70:
            depth_text = "Medium"
            depth_color = (0, 255, 255)  # Yellow
        else:
            depth_text = "Close"
            depth_color = (0, 255, 0)  # Green
            
        # Display depth text and value
        cv2.putText(
            frame, 
            f"Hand Depth: {depth_text} ({normalized_depth:.1f}%)",
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

    def toggle_features(self, use_threshold_adaptation=None, use_temporal_filtering=None):
        """
        Toggle the threshold adaptation and temporal filtering features on or off

        Args:
            use_threshold_adaptation: Boolean to enable/disable threshold adaptation, or None to leave unchanged
            use_temporal_filtering: Boolean to enable/disable temporal filtering, or None to leave unchanged
        """
        if use_threshold_adaptation is not None:
            self.use_threshold_adaptation = use_threshold_adaptation

        if use_temporal_filtering is not None:
            self.use_temporal_filtering = use_temporal_filtering

        # Clear gesture history when turning off temporal filtering
        if use_temporal_filtering is False:
            for gesture_name in self.gesture_history:
                self.gesture_history[gesture_name].clear()
