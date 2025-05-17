import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns
from scipy.signal import savgol_filter
import os
from datetime import datetime


class TemporalFilteringAnalyzer:
    def __init__(self, input_file, history_size=5, confidence_threshold=0.6):
        """
        Initialize the temporal filtering analyzer

        Args:
            input_file: Path to the CSV file with raw gesture detection data
            history_size: Size of the history buffer for temporal filtering
            confidence_threshold: Threshold for binary decision from confidence score
        """
        self.input_file = input_file
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold

        # Load data from CSV
        self.df = pd.read_csv(input_file)

        # Convert gesture to binary values for numerical processing
        self.gesture_mapping = {}
        self.reverse_mapping = {}
        self.detection_results = {}

        # Initialize output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"temporal_analysis_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess_data(self):
        """Preprocess the input data for analysis"""
        # Check if data is already processed
        if 'gesture_binary' in self.df.columns:
            return

        # Generate unique mapping for gestures
        unique_gestures = self.df['gesture'].unique()
        for i, gesture in enumerate(unique_gestures):
            if gesture.lower() == 'null':
                # Use 0 for 'null' gesture (no gesture)
                self.gesture_mapping[gesture] = 0
            else:
                # Use positive integers for actual gestures
                self.gesture_mapping[gesture] = i + 1

        # Create reverse mapping for display
        self.reverse_mapping = {v: k for k, v in self.gesture_mapping.items()}

        # Create binary columns for each gesture
        self.df['gesture_binary'] = self.df['gesture'].map(
            self.gesture_mapping)

        print("Gesture mapping:")
        for gesture, value in self.gesture_mapping.items():
            print(f"  {gesture} → {value}")

        # Check if we have timestamps that make sense
        if self.df['timestamp'].min() < 0 or self.df['timestamp'].max() > 1000:
            # If timestamps don't look like seconds, convert to relative time
            self.df['timestamp'] = self.df['timestamp'] - \
                self.df['timestamp'].min()

        print(f"Processed {len(self.df)} data points")

    def apply_temporal_filtering(self):
        """Apply temporal filtering to the raw detection data"""
        # Initialize data structure for filtered results
        filtered_confidence = np.zeros(len(self.df))

        # Initialize separate history buffers for each gesture
        gesture_history = {}
        unique_gestures = list(self.gesture_mapping.keys())
        for gesture in unique_gestures:
            if gesture.lower() != 'null':  # Skip 'null' gesture
                gesture_history[gesture] = deque(maxlen=self.history_size)

        # Process each frame
        for i, row in self.df.iterrows():
            current_gesture = row['gesture']

            # Update history for each gesture (1 if current gesture matches, 0 otherwise)
            for gesture in gesture_history:
                is_active = 1 if current_gesture == gesture else 0
                gesture_history[gesture].append(is_active)

            # Calculate confidence for current gesture
            if current_gesture.lower() != 'null' and len(gesture_history[current_gesture]) > 0:
                # Apply weighted average (more recent detections have higher weight)
                weights = np.linspace(0.5, 1.0, len(
                    gesture_history[current_gesture]))
                confidence = np.average(
                    gesture_history[current_gesture],
                    weights=weights
                )
                filtered_confidence[i] = confidence

        # Add filtered results to dataframe
        self.df['filtered_confidence'] = filtered_confidence
        self.df['filtered_decision'] = (
            filtered_confidence >= self.confidence_threshold).astype(int)

        # Store a binary column for each unique gesture
        for gesture in unique_gestures:
            if gesture.lower() != 'null':
                col_name = f"{gesture}_raw"
                self.df[col_name] = (self.df['gesture'] == gesture).astype(int)

                col_name = f"{gesture}_filtered"
                self.df[col_name] = ((self.df['gesture'] == gesture) &
                                     (self.df['filtered_confidence'] >= self.confidence_threshold)).astype(int)

    def analyze_stability(self):
        """Analyze stability of filtered vs raw detection"""
        unique_gestures = [
            g for g in self.gesture_mapping.keys() if g.lower() != 'null']

        results = []

        for gesture in unique_gestures:
            # Calculate stability metrics
            raw_col = f"{gesture}_raw"
            filtered_col = f"{gesture}_filtered"

            # Skip if we don't have any detections of this gesture
            if sum(self.df[raw_col]) == 0:
                continue

            # Calculate transitions (changes from 0->1 or 1->0)
            raw_transitions = sum(abs(self.df[raw_col].diff().fillna(0)))
            filtered_transitions = sum(
                abs(self.df[filtered_col].diff().fillna(0)))

            # Calculate stability percentage (fewer transitions = more stable)
            if raw_transitions > 0:
                stability_improvement = (
                    1 - filtered_transitions / raw_transitions) * 100
            else:
                stability_improvement = 0

            # Store results
            results.append({
                'Gesture': gesture,
                'Raw Transitions': raw_transitions,
                'Filtered Transitions': filtered_transitions,
                'Stability Improvement': f"{stability_improvement:.2f}%"
            })

        self.stability_results = pd.DataFrame(results)
        print("\nStability Analysis:")
        print(self.stability_results)

    def plot_all_charts(self):
        """Generate all analysis charts"""
        self.plot_time_series()
        self.plot_gesture_distribution()
        self.plot_transitions()
        self.plot_confidence_distribution()
        self.save_results()

    def plot_time_series(self):
        """Plot time series data showing raw vs filtered detection"""
        unique_gestures = [
            g for g in self.gesture_mapping.keys() if g.lower() != 'null']

        plt.figure(figsize=(14, 8))

        # Plot each gesture
        for gesture in unique_gestures:
            raw_col = f"{gesture}_raw"
            filtered_col = f"{gesture}_filtered"

            # Skip if we don't have any detections of this gesture
            if raw_col not in self.df or sum(self.df[raw_col]) == 0:
                continue

            # Plot with slight offset for visibility
            offset = 0.1 * self.gesture_mapping[gesture]
            plt.plot(self.df['timestamp'], self.df[raw_col] + offset, 'o-',
                     markersize=4, label=f"{gesture} (Raw)", alpha=0.5)
            plt.plot(self.df['timestamp'], self.df[filtered_col] + offset, 'o-',
                     markersize=4, label=f"{gesture} (Filtered)")

        plt.xlabel('Time')
        plt.ylabel('Gesture State (with offset)')
        plt.title('Temporal Filtering: Raw vs Filtered Detection Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/time_series.png", dpi=300)

    def plot_confidence_distribution(self):
        """Plot distribution of confidence scores"""
        plt.figure(figsize=(10, 6))

        # Filter non-zero confidence scores
        confidence_vals = self.df['filtered_confidence'][self.df['filtered_confidence'] > 0]

        if len(confidence_vals) > 0:
            sns.histplot(confidence_vals, bins=20, kde=True)
            plt.axvline(x=self.confidence_threshold, color='r', linestyle='--',
                        label=f'Threshold ({self.confidence_threshold})')

            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Confidence Scores')
            plt.legend()

            # Save the figure
            plt.tight_layout()
            plt.savefig(
                f"{self.output_dir}/confidence_distribution.png", dpi=300)

    def plot_gesture_distribution(self):
        """Plot distribution of gestures in the dataset"""
        plt.figure(figsize=(10, 6))

        # Count occurrences of each gesture
        gesture_counts = self.df['gesture'].value_counts()

        # Create bar chart
        sns.barplot(x=gesture_counts.index, y=gesture_counts.values)
        plt.xlabel('Gesture')
        plt.ylabel('Count')
        plt.title('Distribution of Gestures in Dataset')

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45 if len(gesture_counts) > 4 else 0)

        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/gesture_distribution.png", dpi=300)

    def plot_transitions(self):
        """Plot transitions for each gesture (raw vs filtered)"""
        unique_gestures = [
            g for g in self.gesture_mapping.keys() if g.lower() != 'null']

        plt.figure(figsize=(10, 6))

        raw_transitions = []
        filtered_transitions = []
        gestures = []

        for gesture in unique_gestures:
            raw_col = f"{gesture}_raw"
            filtered_col = f"{gesture}_filtered"

            # Skip if we don't have any detections of this gesture
            if raw_col not in self.df or sum(self.df[raw_col]) == 0:
                continue

            # Calculate transitions (changes from 0->1 or 1->0)
            raw_trans = sum(abs(self.df[raw_col].diff().fillna(0)))
            filtered_trans = sum(abs(self.df[filtered_col].diff().fillna(0)))

            gestures.append(gesture)
            raw_transitions.append(raw_trans)
            filtered_transitions.append(filtered_trans)

        # Create grouped bar chart
        x = np.arange(len(gestures))
        width = 0.35

        plt.bar(x - width/2, raw_transitions, width, label='Raw Detection')
        plt.bar(x + width/2, filtered_transitions,
                width, label='Filtered Detection')

        plt.xlabel('Gesture')
        plt.ylabel('Number of Transitions')
        plt.title('Stability Comparison: Raw vs Filtered Detection')
        plt.xticks(x, gestures)
        plt.legend()

        # Add percentage improvement labels
        for i in range(len(gestures)):
            if raw_transitions[i] > 0:
                improvement = (
                    1 - filtered_transitions[i] / raw_transitions[i]) * 100
                plt.text(i, max(raw_transitions[i], filtered_transitions[i]) + 0.5,
                         f"{improvement:.1f}%", ha='center')

        # Save the figure
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/transitions_comparison.png", dpi=300)

    def save_results(self):
        """Save analysis results to CSV"""
        # Save processed data
        self.df.to_csv(f"{self.output_dir}/processed_data.csv", index=False)

        # Save stability results
        if hasattr(self, 'stability_results'):
            self.stability_results.to_csv(
                f"{self.output_dir}/stability_results.csv", index=False)

        print(f"\nResults saved to directory: {self.output_dir}")

    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print(f"Analyzing file: {self.input_file}")
        self.preprocess_data()
        self.apply_temporal_filtering()
        self.analyze_stability()
        self.plot_all_charts()
        print("Analysis complete!")


# Script to run the analysis
if __name__ == "__main__":
    # Get the input file name
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Default input file
        input_file = "results\saved\16-5-25\temporal_test_20250515_235005 (without_filtering).csv"

    # Create analyzer with default parameters
    analyzer = TemporalFilteringAnalyzer(
        input_file=input_file,
        history_size=5,        # Default history buffer size
        confidence_threshold=0.6  # Default confidence threshold
    )

    # Run full analysis
    analyzer.run_full_analysis()
