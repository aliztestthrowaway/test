# Import libraries
from sklearn.linear_model import LogisticRegression as lr
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import pandas as pd
import numpy as np

# Read-in the sample data
dfData = pd.read_csv('data.csv')

# Separate the Predictor and Predicted variables
lX = dfData.iloc[:, :-1].to_numpy()
lY = dfData.iloc[:, -1].to_numpy()


# Class to calculate the Classification Threshold using Gini Impurity
class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    # Nothing to set when initializing the class
    def __init__(self):
        pass
    
    # Calculate and return the Threshold based on the Gini Impurity
    def fit_transform(self, Y):
        
        iRecords = len(Y)
        lUnique, lCounts = np.unique(Y, return_counts = True)
        iThreshold = sum(lCounts / iRecords * (1 - lCounts / iRecords))
                    
        self.iThreshold = iThreshold
        
        return iThreshold


# Class to classify the data with a Logistic Regression and custom Threshold
class custom_estimator(BaseEstimator, ClassifierMixin):  
    # Set the Threashold from the imput
    def __init__(self, iThreshold):
        self.iThreshold = iThreshold
    
    # Fit a logistical regression onto the input data
    def fit(self, X, y = None):
        
        modelLr = lr(random_state = 0, 
                     solver = 'liblinear',
                     multi_class = 'ovr'
                     )
        modelLr.fit(X, y)
        
        self.modelLr = modelLr       
        
        return None
    
    # Predict the classes based on the input threashold parameter
    def predict(self, X):
        
        lPreds = self.modelLr.predict_proba(X)[:, 0]
        lPreds[lPreds < self.iThreshold] = 0
        lPreds[lPreds >= self.iThreshold] = 1
        
        self.lPreds = lPreds
        
        return lPreds


if __name__ == "__main__":
    # Threshold calculation
    iThreashold = ThresholdBinarizer().fit_transform(lY)    
    
    # Create the custom estimator
    glm = custom_estimator(iThreashold)
    
    # Fit the model
    glm.fit(lX, lY)
    
    # Make prediction
    lPreds = glm.predict(lX)


