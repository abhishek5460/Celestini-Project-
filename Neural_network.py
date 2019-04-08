NEURAL NETWORK


import pandas as pd
dataset = pd.read_csv('zoo.csv')
X = dataset.iloc[:,1:17].values
y = dataset.iloc[:, 17].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
y_b= np_utils.to_categorical(y_train)   
classifier = Sequential()   
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = 16))
classifier.add(Dense(output_dim = 40, init = 'uniform', activation = 'relu'))    
classifier.add(Dense(output_dim = 40, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'softmax'))   
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])    
classifier.fit(X_train, y_b, batch_size = 5, nb_epoch = 100)
y_pred = classifier.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)      
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
from sklearn.metrics import classification_report
target_names = ['Class-1','Class-2', 'Class-3 ','Class-4','Class-5','Class-6','Class-7']
print(classification_report(y_test, y_pred, target_names=target_names))

