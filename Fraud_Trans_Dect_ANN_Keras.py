""" ----------------------Fraud Transaction Detection with Artificial Neuron Network (ANN) -------------------------"""

# import required library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#%%
# Load creditcard dataset from Kaggle [link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]

credit_card_data = pd.read_csv('C:\\Users\\Salim Raza\\\Desktop\\UTEP Research\\Code\\Fraud Transaction Detection (FTD)\\creditcard.csv')
print(credit_card_data.head()) # First 5 rows of the dataset
print(credit_card_data.tail()) # Last 5 rows of the dataset

#%% 
# Shows sahape of dataset 

credit_card_data.info() # Dataset information
print(credit_card_data.isnull()) # Checking missing values as a True and False
print(credit_card_data.isnull().sum()) #Suming  the True values (Missing values) in each column  
print(credit_card_data['Class'].value_counts()) # Showing distribution of transactions in terms of Class

#%% 
# Separating the data for analysis

legit = credit_card_data[credit_card_data['Class'] == 0]
fraud = credit_card_data[credit_card_data['Class'] == 1]
print(legit.shape)
print(fraud.shape)

#%% 
# Preprocessing dataset to fit them with the model (1)

print(legit['Amount'].describe()) # Statistical measures of legit the dataset
print(fraud['Amount'].describe()) # Statistical measures of fraud the dataset
print(credit_card_data.groupby('Class').mean())  # Showig mean values for both class for each feature

#%% 
#Preprocessing dataset to fit them with the model (2)

legit_sample = legit.sample(n=492)  # Select 492 random rows from the legit, this is equal to fraud dataset
new_dataset = pd.concat([legit_sample, fraud], axis=0) # Combines the two datasets (legit+fraud) vertically

print(new_dataset.head())
print(new_dataset.tail())

print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X)
print(Y)

#%% 
#Preprocessing dataset to fit them with the model (3)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


#%%
# Create a Sequential ANN Nodel (Feedforward Neural Network)

model = Sequential()
model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))

#%%
# Compile and Train the model 

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10000, batch_size=32)

#%%
# Prediction and Accuracy  on Training Data

y_train_pred = model.predict(X_train)
y_train_pred = (y_train_pred > 0.5).astype(int)
training_data_accuracy = accuracy_score(y_train, y_train_pred)
print('Accuracy on Training data:', training_data_accuracy)

#%%
# Prediction and Accuracy on Testing Data

y_test_pred = model.predict(X_test)
y_test_pred = (y_test_pred > 0.5).astype(int)
test_data_accuracy = accuracy_score(y_test, y_test_pred)
print('Accuracy on Test Data:', test_data_accuracy)

""" ------------------------------------------- End-----------------------------------------------------------"""