'''
#1: Adding batch od data in NN
import numpy as np
input=[[1,2,3,4],
       [2,3,4,5],
       [5,6,7,8]] # 3 batches of data
weights=[[1,20,-3,6],
        [2,4,-5,1],
        [3,2,5,-1]]
biases=[-2,3,-1]

    
''
#output=(np.dot(weights,input))+biases
   - This gives shape error since matrix dimention of both input and weights are 3x4.
   - by doing np.array python list getting converted to numpy array where Transpose can be applied.
''

output=np.dot(input,np.array(weights).T)+biases #solution: Transpose '.T' does transpose
print(output)

'''


#2: Manually adding layers for forward pas
import numpy as np

input=[[1,2,3,4],
       [2,3,4,5],
       [5,6,7,8]]
weights1=[[1,20,-3,6],
        [2,4,-5,1],
        [3,2,5,-1]]
weights2=[[1.1,2.2,3.1],
        [5.1,1.1,-2.1],
        [3.1,1,-3.1]]
biases1=[-2,3,-1]
biases2=[-300,100,-400]

layer1output=np.dot(input,np.array(weights1).T)+biases1
layer2output=np.dot(layer1output, np.array(weights2).T)+biases2


print(layer2output)


