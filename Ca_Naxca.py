import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats

import matplotlib.pyplot as plt

class regression_report:
    def __init__(self, y_test, y_pred, IDs): #IDs es lista t.q. IDs[x] es el ID del sample con label y_test[x] y predicción y_pred[x]
        if len(y_test) != len(y_pred):
            raise Exception("Test and Prediction samples ammount does not match!")
        if len(y_test) != len(IDs):
            raise Exception("ID's ammount does not match with test and prediction!")
        
        self.IDs = IDs
        self.length = len(y_test)

        self.y_test = y_test
        self.y_pred = y_pred
        #self.diferencias = [abs(y_test[x] - y_pred[x]) for x in range(self.length)))]

    def display(self):
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)

        r_squared = r2_score(self.y_test, self.y_pred)
        spearman = stats.spearmanr(self.y_test, self.y_pred)
        pearson = stats.pearsonr(self.y_test, self.y_pred)
        kendalltau = stats.kendalltau(self.y_test, self.y_pred)

        #mdae = median_absolute_error(y_test, y_pred)
        #mdape = ((pd.Series(y_test) - pd.Series(y_pred)) / pd.Series(y_test)).abs().median()    
        #mape = mean_absolute_percentage_error(y_test, y_pred)
        
        df = pd.DataFrame.from_dict({
                                #"Median Absolute Error": [mdae],
                                #"Mean Absolute Percentage Error": [mape],
                                #"MDAPE": [mdape],
                                 "Mean Absolute Error": [mae],
                                 "Mean Squared Error": [mse],
                                 "R2 score": [r_squared],
                                 "Spearman": [spearman],
                                 "Pearson": [pearson],
                                 "Kendall-Tau": [kendalltau]})
        return df


    def plot(self, path):
        cuantos = range(len(self.IDs))

        plt.figure(figsize=(10, 6))

        plt.scatter(cuantos, self.y_test, color='blue', label='Real Values', marker='o')
        plt.scatter(cuantos, self.y_pred, color='red', label='Pred. Values', marker='x')

        plt.xticks(cuantos, self.IDs, rotation=45)
        plt.xlabel('IDs')
        plt.ylabel('Labels')
        plt.title('Predicted vs Real values')
        plt.legend()

        plt.tight_layout()
        plt.savefig(path)
        
        plt.show()
