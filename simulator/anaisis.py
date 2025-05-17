import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = r"C:\Users\farhat\Desktop\TA\test-hand-modul-TA\simulator\scenario_1_noise_.csv"
data = pd.read_csv(file_path)

# Check if required columns exist
required_columns = ['Filtered_Decision',
                    'Filtered_Confidence', 'Raw_Detection', 'Frame']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Dataset columns: {list(data.columns)}")
    raise KeyError(
        f"The following required columns are missing from the dataset: {missing_columns}")

# Calculate transitions in Filtered_Decision
data['Decision_Change'] = data['Filtered_Decision'].diff().abs()
transitions = data['Decision_Change'].sum()

# Calculate statistics for Filtered_Confidence
confidence_stats = data.groupby('Raw_Detection')[
    'Filtered_Confidence'].agg(['mean', 'std'])

# Calculate false positives and false negatives
false_positives = len(
    data[(data['Raw_Detection'] == 0) & (data['Filtered_Decision'] == 1)])
false_negatives = len(
    data[(data['Raw_Detection'] == 1) & (data['Filtered_Decision'] == 0)])
total_frames = len(data)
false_positive_rate = false_positives / total_frames * 100
false_negative_rate = false_negatives / total_frames * 100

# Plot Filtered_Confidence over time
plt.figure(figsize=(10, 6))
plt.plot(data['Frame'], data['Filtered_Confidence'],
         label='Filtered Confidence')
plt.scatter(data['Frame'], data['Filtered_Decision'],
            color='red', label='Filtered Decision', alpha=0.5)
plt.xlabel('Frame')
plt.ylabel('Filtered Confidence')
plt.title('Filtered Confidence and Decision Over Time')
plt.legend()
plt.show()

# Print results
print(f"Number of transitions in Filtered_Decision: {transitions}")
print(f"Confidence stats by Raw_Detection:\n{confidence_stats}")
print(f"False Positive Rate: {false_positive_rate:.2f}%")
print(f"False Negative Rate: {false_negative_rate:.2f}%")
