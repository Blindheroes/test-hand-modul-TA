import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
import csv


class GestureSimulator:
    def __init__(self, total_frames=200, noise_probability=0.2, gesture_change_probability=0.05):
        """
        Initialize the gesture simulator

        Args:
            total_frames: Number of frames to simulate
            noise_probability: Probability of noise in any given frame
            gesture_change_probability: Probability of changing the true gesture state
        """
        self.total_frames = total_frames
        self.noise_probability = noise_probability
        self.gesture_change_probability = gesture_change_probability

        # Ground truth for the gesture (1=active, 0=inactive)
        self.true_gesture = np.zeros(total_frames, dtype=int)

        # Noisy raw detections (before filtering)
        self.raw_detections = np.zeros(total_frames, dtype=int)

        # Filtered detections with temporal filtering
        self.filtered_detections = np.zeros(total_frames, dtype=float)

        # History buffer for temporal filtering
        self.history_buffer = deque(maxlen=5)

        # Thresholds for detection
        self.detection_threshold = 0.6

    def generate_true_gesture_data(self):
        """Generate the ground truth gesture states with realistic transitions"""
        # Start with gesture inactive
        current_state = 0

        for i in range(self.total_frames):
            # Randomly change gesture state with low probability
            if random.random() < self.gesture_change_probability:
                current_state = 1 - current_state  # Toggle state

            self.true_gesture[i] = current_state

            # Ensure gestures last for at least 10 frames (realistic user behavior)
            if i > 0 and self.true_gesture[i] != self.true_gesture[i-1]:
                # Keep the same state for next 10 frames minimum
                end_idx = min(i + 10, self.total_frames)
                self.true_gesture[i:end_idx] = current_state

        return self.true_gesture

    def generate_noisy_detections(self):
        """Generate noisy raw detection data based on true gesture states"""
        for i in range(self.total_frames):
            # Add noise based on probability
            if random.random() < self.noise_probability:
                # Flip the true state to simulate noise
                self.raw_detections[i] = 1 - self.true_gesture[i]
            else:
                # No noise, detection matches true state
                self.raw_detections[i] = self.true_gesture[i]

        return self.raw_detections

    def apply_temporal_filtering(self):
        """Apply temporal filtering to the raw detections"""
        for i in range(self.total_frames):
            # Add current detection to history buffer
            self.history_buffer.append(self.raw_detections[i])

            if len(self.history_buffer) > 0:
                # Apply weighted average (more recent detections have higher weight)
                weights = np.linspace(0.5, 1.0, len(self.history_buffer))
                confidence = np.average(self.history_buffer, weights=weights)

                # Store the confidence score
                self.filtered_detections[i] = confidence
            else:
                self.filtered_detections[i] = self.raw_detections[i]

        return self.filtered_detections

    def get_filtered_decisions(self):
        """Convert confidence values to binary decisions using threshold"""
        return (self.filtered_detections >= self.detection_threshold).astype(int)

    def calculate_accuracy(self, detections):
        """Calculate accuracy compared to ground truth"""
        correct = np.sum(detections == self.true_gesture)
        return correct / self.total_frames * 100

    def save_to_csv(self, filename="gesture_simulation_data.csv"):
        """Save simulation data to CSV file"""
        filtered_decisions = self.get_filtered_decisions()

        # Calculate metrics
        raw_accuracy = self.calculate_accuracy(self.raw_detections)
        filtered_accuracy = self.calculate_accuracy(filtered_decisions)

        # Create DataFrame
        df = pd.DataFrame({
            'Frame': range(1, self.total_frames + 1),
            'True_Gesture': self.true_gesture,
            'Raw_Detection': self.raw_detections,
            'Filtered_Confidence': self.filtered_detections,
            'Filtered_Decision': filtered_decisions
        })

        # Save to CSV
        df.to_csv(filename, index=False)

        print(f"Data saved to {filename}")
        print(f"Raw Detection Accuracy: {raw_accuracy:.2f}%")
        print(f"Filtered Detection Accuracy: {filtered_accuracy:.2f}%")

        return df

    def run_simulation(self):
        """Run the complete simulation pipeline"""
        self.generate_true_gesture_data()
        self.generate_noisy_detections()
        self.apply_temporal_filtering()
        return self.save_to_csv()

    def plot_results(self):
        """Plot the simulation results for visualization"""
        filtered_decisions = self.get_filtered_decisions()

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 10), sharex=True)

        # Ground truth
        ax1.plot(range(self.total_frames), self.true_gesture,
                 'g-', label='Ground Truth')
        ax1.set_ylabel('Gesture State')
        ax1.set_title('Ground Truth Gesture')
        ax1.legend()
        ax1.set_ylim(-0.1, 1.1)

        # Raw detections with noise
        ax2.plot(range(self.total_frames), self.raw_detections,
                 'r-', label='Raw Detection')
        ax2.set_ylabel('Detection State')
        ax2.set_title('Raw Detections (with noise)')
        ax2.legend()
        ax2.set_ylim(-0.1, 1.1)

        # Filtered decisions
        ax3.plot(range(self.total_frames), self.filtered_detections,
                 'b-', alpha=0.5, label='Confidence Score')
        ax3.plot(range(self.total_frames), filtered_decisions,
                 'b--', label='Filtered Decision')
        ax3.axhline(y=self.detection_threshold, color='gray',
                    linestyle='--', alpha=0.7, label='Threshold')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Confidence / Decision')
        ax3.set_title('Temporal Filtering Results')
        ax3.legend()
        ax3.set_ylim(-0.1, 1.1)

        plt.tight_layout()
        plt.savefig('gesture_simulation_results.png', dpi=300)
        plt.show()

# Function to generate multiple simulation scenarios


def generate_multiple_simulations(scenarios):
    """
    Generate multiple simulation scenarios with different parameters

    Args:
        scenarios: List of dictionaries with simulation parameters
    """
    results = []

    for i, params in enumerate(scenarios):
        print(
            f"\nRunning simulation {i+1}/{len(scenarios)} with parameters: {params}")

        simulator = GestureSimulator(
            total_frames=params.get('total_frames', 200),
            noise_probability=params.get('noise_probability', 0.2),
            gesture_change_probability=params.get(
                'gesture_change_probability', 0.05)
        )

        # Run simulation
        df = simulator.run_simulation()

        # Plot results if specified
        if params.get('plot', False):
            simulator.plot_results()

        # Calculate accuracy
        filtered_decisions = simulator.get_filtered_decisions()
        raw_accuracy = simulator.calculate_accuracy(simulator.raw_detections)
        filtered_accuracy = simulator.calculate_accuracy(filtered_decisions)

        # Add to results
        results.append({
            'Scenario': i+1,
            'Parameters': params,
            'Raw_Accuracy': raw_accuracy,
            'Filtered_Accuracy': filtered_accuracy,
            'Improvement': filtered_accuracy - raw_accuracy
        })

        # Save with specific filename
        scenario_name = f"scenario_{i+1}_noise_{params.get('noise_probability', 0.2)}.csv"
        simulator.save_to_csv(scenario_name)

    # Create summary CSV
    with open('simulation_summary.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Scenario', 'Noise_Probability', 'Gesture_Change_Probability',
                        'Raw_Accuracy', 'Filtered_Accuracy', 'Improvement'])

        for res in results:
            writer.writerow([
                res['Scenario'],
                res['Parameters'].get('noise_probability', 0.2),
                res['Parameters'].get('gesture_change_probability', 0.05),
                f"{res['Raw_Accuracy']:.2f}%",
                f"{res['Filtered_Accuracy']:.2f}%",
                f"{res['Improvement']:.2f}%"
            ])

    print("\nSimulation summary saved to simulation_summary.csv")


# Run the simulation with different noise levels
if __name__ == "__main__":
    # Define different simulation scenarios
    scenarios = [
        # {
        #     'total_frames': 200,
        #     'noise_probability': 0.1,  # Low noise
        #     'gesture_change_probability': 0.05,
        #     'plot': True
        # },
        # {
        #     'total_frames': 200,
        #     'noise_probability': 0.3,  # Medium noise
        #     'gesture_change_probability': 0.05,
        #     'plot': True
        # },
        {
            'total_frames': 600,
            'noise_probability': 0.5,  # High noise
            'gesture_change_probability': 0.05,
            'plot': True
        }
    ]

    generate_multiple_simulations(scenarios)

    # Additionally, run a single detailed simulation for visualization
    print("\nRunning detailed simulation for visualization...")
    sim = GestureSimulator(total_frames=200, noise_probability=0.3)
    sim.run_simulation()
    sim.plot_results()
