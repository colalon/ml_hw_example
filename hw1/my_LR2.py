import numpy as np


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
    def __init__(self, learning_rate = 1e-4):
        self.learning_rate = learning_rate
    def norm_mean_std(self,x):
        self.x_mean=np.zeros((x.shape[1]))
        self.x_std=np.zeros((x.shape[1]))
        for i in range (x.shape[1]):
            self.x_mean[i]=x[:,i].mean()
            self.x_std[i]=x[:,i].std() + 0.00001
    def normalize(self,x):
        self.x_norm = np.zeros(x.shape)
        for i in range (x.shape[1]):
            self.x_norm[:,i] = (x[:,i]-self.x_mean[i])/self.x_std[i]
            
    def predict(self,x):
        self.normalize(x)
        
        return np.matmul(self.x_norm,self.weights) + self.bias
    
    def score(self,x,y):
        self.normalize(x)
        
        all_score = y - (np.matmul(self.x_norm,self.weights) + self.bias)
        all_score = all_score * all_score
        
        return all_score.mean()
    def get_Params(self):
        return self.weights,self.bias
    
    def fit(self,x,y,iterstions=2000):
        self.norm_mean_std(x)
        self.normalize(x)
        self.weights = np.random.normal(0,0.1,(x.shape[1],1))*0.5
        self.bias = 0
        
        for it in range (iterstions):
            for i in range (x.shape[1]):
                weights_diff = error_diff(self.x_norm,y,self.weights,self.bias,i)*self.weights[i]
                self.weights[i]= self.weights[i] - self.learning_rate * weights_diff
                
            bias_diff = error_diff_bias(self.x_norm,y,self.weights,self.bias)
            self.bias= self.bias - self.learning_rate * bias_diff
            
            if it%1000 == 0:
                print (self.score(x,y))
        
                