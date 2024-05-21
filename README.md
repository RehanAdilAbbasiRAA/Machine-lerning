import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# here WE HAVE LOAD THE DATA IN LOCAL FORM BY FIRST PUT IT IN LIST. 
STUDENT=9.2
data = {
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86]
}
df = pd.DataFrame(data)

plt.scatter(df['Hours'], df['Scores'], color='blue')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.show()

# Split the data into training and testing sets
X = df[['Hours']]
y = df['Scores']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# WE TRAIN OUR MODEL HERE ,SO IT CAN PREDICT THE OUTCOME
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make a prediction for 9.2 hours
hours = pd.DataFrame({'Hours': [9.2]})
predicted_score = regressor.predict(hours)
print(f"Predicted score for a student studying {STUDENT} hours a day: {predicted_score[0]:.2f}")





