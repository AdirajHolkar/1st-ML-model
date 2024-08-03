#import of all the library
import pandas as pd    #for data set
from sklearn.model_selection import train_test_split  #for spliting the data
from sklearn.ensemble import RandomForestClassifier   #Machine learning model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  #Some metrix

# Load the dataset
df = pd.read_csv('online_payment_fraud_dataset.csv')   

# Loading / operating on the data;
print(df.describe())
print(df.isnull().sum())  

#----Processing of the data-----

# Convert the 'type' column to categorical values
df['type'] = pd.Categorical(df['type']).codes

# Convert the 'isFraud' column to binary values (0/1)
df['isFraud'] = df['isFraud'].map({True: 1, False: 0}) 


# Spliting the data into training and testing sets (80% for training and 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(df.drop('isFraud', axis=1), df['isFraud'], test_size=0.2, random_state=42)  

# Using the random forest model 

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train) 


y_pred = rfc.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred)) 

 
import pickle
with open('online_payment_fraud_model.pkl', 'wb') as f:
    pickle.dump(rfc, f)


with open('online_payment_fraud_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

new_data = pd.DataFrame({'step': [1], 'type': [1], 'amount': [100], 'nameOrig': ['John'], 'oldbalanceOrg': [1000], 'newbalanceOrig': [900], 'nameDest': ['Jane'], 'oldbalanceDest': [500], 'newbalanceDest': [600]})
new_prediction = loaded_model.predict(new_data)
print("Prediction:", new_prediction)