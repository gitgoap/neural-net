import numpy as np
'''
#This is same code as p1. np.dot do element wise multiplication then addition (automate code in p1_neuron-code.py) 
input=[1,2,3,4]
weight=[0.1,0.2,0.3,0.4]
bias=3
output=np.dot(input,weight)+bias
print(output)

'''

# array are homologous a=[[1,2,3], [2,3]] not acceptable, a=[[1,2,3], [4,5,6]] acceptable

input=[1,2,3,4]
weights=[[1,20,-3,6],
        [2,4,-5,1],
        [3,2,5,-1]]

biases=[-2,3,-1]

output=np.dot(weights,input)+biases #order of parameters in dot matter, otherwise gives array shape error

print(output)


'''
np.dot(weight,input) equivalent to [np.dot(weight[0],input), np.dot(weight[1],input), np.dot(weight[2],input)]
'''