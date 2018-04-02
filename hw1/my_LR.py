import numpy as np


def error(x,y,weight,bias):
    #n = y.shape[0]
    error_all = y - (np.matmul(x,weight) + bias)
    error_all_squre = error_all * error_all
    error_mean = error_all_squre.mean()
    return error_mean

def error_diff(x,y,weight,bias,num):
    error_all = y - (np.matmul(x,weight) + bias)
    diff_all = error_all * x[:,num:num+1]
    diff_mean = -2 * diff_all.mean()
    return diff_mean
    
def error_diff_bias(x,y,weight,bias):
    error_all = y - (np.matmul(x,weight) + bias)
    diff_all = error_all
    diff_mean = -2 * diff_all.mean()
    return diff_mean  

class my_LR():
    def __init__(self, learning_rate = 1e-6):
        self.learning_rate = learning_rate
        
    def get_Params(self):
        return self.weights,self.bias
    
    def predict(self,x):
        return np.matmul(x,self.weights) + self.bias
    
    def score(self,x,y):
        return error(x,y,self.weights,self.bias)
    
    
    def fit(self,x,y,iterstions=50000):
        self.weights = np.random.normal(0,0.2,(x.shape[1],1))*1
        self.bias = 0
        
        for it in range (iterstions):
            for i in range (x.shape[1]):
                batch = np.random.randint(x.shape[0],size=(100))
                weights_diff = error_diff(x[batch],y[batch],self.weights,self.bias,i)
                self.weights[i] = self.weights[i] - self.learning_rate * weights_diff
            self.bias = self.bias -  self.learning_rate * error_diff_bias(x[batch],y[batch],self.weights,self.bias) * 1
            
            
            if it % 5000 == 0:
                self.error_predict = error(x,y,self.weights,self.bias)
                print ('iterations',it,'cost',self.error_predict)
                
                