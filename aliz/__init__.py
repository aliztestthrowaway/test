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
    # Set the predicted and origianl classes from the input parameters
    def __init__(self, P, Y):
        self.P = np.reshape(P, (len(P), 1))
        self.Y = np.reshape(Y, (len(Y), 1))
    
    # Optimize and return the threshold based on the Gini Impurity
    def fit(self, step):
        iThreshold = 0
        fGini = 1.0
        iUpper = 1 + step
        
        for iTh in np.arange(0, iUpper, step):
            lP = np.copy(self.P)
            lY = np.copy(self.Y)

            lP[lP < iTh] = 0
            lP[lP >= iTh] = 1
            lR = lP - lY
            
            lR = np.abs(lR)
            iRecords = len(lR)
            lUnique, lCounts = np.unique(lR, return_counts = True)
            
            fGiniCurr = sum(lCounts / iRecords * (1 - lCounts / iRecords))
            
            if fGiniCurr < fGini:
                fGini = fGiniCurr
                iThreshold = iTh
        
        self.iThreshold = iThreshold
        
        print('Optimal threshold: %f; Gini Impurity: %f' % (iThreshold, fGini))
        
        return iThreshold


# Class to classify the data with a Logistic Regression and optimized Threshold
class custom_estimator(BaseEstimator, ClassifierMixin):  
    # Set the step for calculating the threshold from the input parameter
    def __init__(self, step = 0.01):
        self.step = step
    
    # Fit a logistical regression onto the input data
    def fit(self, X, Y = None):
        
        modelLr = lr(random_state = 0, 
                     solver = 'liblinear',
                     multi_class = 'ovr'
                     )
        modelLr.fit(X, Y)
        
        self.modelLr = modelLr       
        
        return None
    
    # Predict the classes based on the optimal threashold
    def predict(self, X):
        
        lPreds = self.modelLr.predict_proba(X)[:, 0]

        # Threshold optimization
        self.iThreshold = ThresholdBinarizer(lPreds, lY).fit(self.step)   
        
        lPreds[lPreds < self.iThreshold] = 0
        lPreds[lPreds >= self.iThreshold] = 1
        
        self.lPreds = lPreds
        
        return lPreds


if __name__ == "__main__":
    
    # Create the custom estimator
    glm = custom_estimator(step = 0.001)
    
    # Fit the model
    glm.fit(lX, lY)
    
    # Make the prediction
    lPreds = glm.predict(lX)
