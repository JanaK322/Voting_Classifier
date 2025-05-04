import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import pickle

# Load your data
df = pd.read_csv(r"C:\Users\janas\Downloads\Telegram Desktop\neo_asteroids_data.csv")

# Create diameter_avg
df['diameter_avg'] = (df['diameter_km_min'] + df['diameter_km_max']) / 2

def remove_outliers_iqr_iterative(df, columns):
    prev_shape = None
    while prev_shape != df.shape:
        prev_shape = df.shape
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Apply to your DataFrame
df = remove_outliers_iqr_iterative(df, ['diameter_avg', 'velocity_km_s', 'miss_distance_km'])


# Prepare features and label
X = df[['velocity_km_s', 'miss_distance_km', 'diameter_avg']]
y = df['hazardous'].astype(int)  # 1 for hazardous, 0 for not

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler =preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Convert back to DataFrame to print after scaling
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# Define base models with default parameters
lr = LogisticRegression()
knn = KNeighborsClassifier()
svm = SVC(probability=True)

# Combine them in a soft voting classifier
voting_d = VotingClassifier(estimators=[
    ('lr', lr),
    ('knn', knn),
    ('svm', svm)
])

# Fit the model
voting_d.fit(X_train, y_train)

# Predict on test set
y_pred = voting_d.predict(X_test)
y_train_pred = voting_d.predict(X_train)

# Save to pickle
pickle.dump(voting_d, open('voting_model.pkl', 'wb'))

