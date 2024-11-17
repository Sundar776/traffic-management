import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Simulate traffic data
data = {
    'vehicle_count': np.random.randint(50, 300, 1000),  # number of vehicles
    'time_of_day': np.random.randint(0, 24, 1000),  # hour of the day (0-23)
    'day_of_week': np.random.randint(0, 7, 1000),  # day of the week (0=Sunday, 6=Saturday)
    'current_signal': np.random.choice(['red', 'green', 'yellow'], 1000),  # current signal status
    'congestion_level': np.random.choice([0, 1], 1000)  # 0 = low, 1 = high
}

df = pd.DataFrame(data)

# Convert categorical data to numerical values
df['current_signal'] = df['current_signal'].map({'red': 0, 'green': 1, 'yellow': 2})

# Step 2: Prepare the data for model training
X = df[['vehicle_count', 'time_of_day', 'day_of_week', 'current_signal']]
y = df['congestion_level']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")  # Output the accuracy of the model

# Step 4: Define the traffic management function
def manage_traffic_signal(vehicle_count, time_of_day, day_of_week, current_signal):
    # Predict congestion level
    congestion_pred = model.predict([[vehicle_count, time_of_day, day_of_week, current_signal]])
    
    # Make a decision based on the prediction
    if congestion_pred == 1:  # If congestion is predicted
        if current_signal == 0:  # if signal is red
            return "Turn signal to green to ease traffic."
        elif current_signal == 2:  # if signal is yellow
            return "Extend yellow duration."
        else:
            return "Keep signal green."
    else:
        if current_signal == 1:  # if signal is green
            return "Consider switching to yellow to manage flow."
        else:
            return "Keep current signal."

# Step 5: Test the function with example inputs
test_result = manage_traffic_signal(vehicle_count=200, time_of_day=17, day_of_week=4, current_signal=0)
print("Traffic Management Decision:", test_result)  # Output the traffic management decision