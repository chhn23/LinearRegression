import numpy as np
class Linear_Regression:
  #initiating the parameters learning_rate, no_of_iterations
  def __init__(self, learning_rate, no_of_iterations):
    self.learning_rate=learning_rate
    self.no_of_iterations=no_of_iterations

  def fit(self,X,Y):
    #no of training examples and number of features
    self.m, self.n = X.shape #no of rows and columns

    #initiating the weight and bias of the model
    self.w=np.zeros(self.n)
    self.b=0
    self.X=X
    self.Y=Y

    #implementing gradient Descent
    for i in range(self.no_of_iterations):
      self.update_weight()

  def update_weight(self):
    Y_prediction=self.predict(self.X)
    #calculate gradient
    dw=-(2*(self.X.T).dot(self.Y-Y_prediction))/self.m
    db= -2*np.sum(self.Y-Y_prediction)/self.m
    #Updating the weights
    self.w=self.w-self.learning_rate*dw
    self.b=self.b-self.learning_rate*db

  def predict(self,X):
    return X.dot(self.w)+self.b