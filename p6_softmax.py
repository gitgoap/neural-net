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
