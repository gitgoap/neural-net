#input --> Exponential --> Softmax Normalization --> Output

#Every thing from scratch
'''
E=2.71828 # import math E=math.e # gives the value of euler constant i.e. 2.71828
X=[4.8,1.21,2.385]
exp_value=[]
normalized_value=[]
sum=0
for value in X:
    exp_value.append(E**value)
    sum=sum+E**value
for final in exp_value:
    normalized_value.append(final/sum)

print(normalized_value)    
print(sum)
'''



#Using Numpy lib
'''
import numpy as np
X=[4.8,1.21,2.385]
exp_value=np.exp(X)
normalized_value=exp_value/np.sum(exp_value)
print(normalized_value)

'''

#For Batches
'''
import numpy as np
X=[[2,3,6],
   [4,1,2],
   [-4,1,5]]
exp_value=np.exp(X)
normalized_value=exp_value/ np.sum(exp_value, axis=1, keepdims=True)
print(normalized_value)

#axis=0 : column; axis=1 : rows
'''

#overflow prevention
'''
 error like: e^1000 gives Runtime warning 
 prevention: final values (v) = ui- max (ui)
             e.g. [2,3,1] after prevention [2-3,3-3,1-3] = [-1,0,-2] 
 prevention doesn't effect final normalized value.             
'''

#full code 

import numpy as np
import math
np.random.seed(0)
X=[[2,3,6],
   [4,1,2],
   [-4,1,5]]


class neural_net:
    def __init__(self,layer_n,nextOFlayer_n):
        self.weights=np.random.randn(layer_n,nextOFlayer_n)
        self.biases=np.zeros((1,nextOFlayer_n))

    def forward(self,input):
        self.output=np.dot(input, self.weights) + self.biases

class activation:
    def relu(self,input):
        self.output_r=np.maximum(0,input)

    def sigmoid(self,input):
        self.output_s=1/(1+np.exp(-input)) 

    def softmax(self,input):
        exp_value=np.exp(input)
        #self.output=np.exp(input)/np.sum((np.exp(input), axis=1, keepdims=True))    # this not working
        self.output=exp_value/ np.sum(exp_value, axis=1, keepdims=True)

first_layer=neural_net(3,3) #layers overview 3x5x4x2
first_layer.forward(X)
first_layer_activ=activation()
first_layer_activ.sigmoid(first_layer.output) #sigmoid for output for second layer applied here

second_layer=neural_net(3,4)
second_layer.forward(first_layer_activ.output_s) 
second_layer_activ=activation()
second_layer_activ.relu(second_layer.output) #relu here


final_layer=neural_net(4,2)
final_layer.forward(second_layer_activ.output_r)
Total_output=activation()
Total_output.softmax(final_layer.output)

print(Total_output.output)




