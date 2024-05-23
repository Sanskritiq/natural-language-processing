from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt

class Evaluate:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.accuracy = self.accuracy()
        self.precision = self.precision()
        self.recall = self.recall()
        self.f1 = self.f1()
        
        
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def precision(self):
        return precision_score(self.y_true, self.y_pred, average='weighted')
    
    def recall(self):
        return recall_score(self.y_true, self.y_pred, average='weighted')
    
    def f1(self):
        return f1_score(self.y_true, self.y_pred, average='weighted')
    