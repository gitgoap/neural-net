import numpy as np
class neuron:
    def __init__(self,weight,bias):
        self.weight=weight
        self.bias=bias
    def feedForward(self,input):
         return sigmoid(np.dot(input,self.weight)+ self.bias)


def sigmoid(x):
    return 1/(1+np.exp(-x))
         
w=np.array([0,1])
i=np.array([2,3])
b=4
first=neuron(w,b)
output=first.feedForward(i)
print(output)




