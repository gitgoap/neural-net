#input --> Exponential --> Normalize --> Output

import math
E=math.e # gived the value of euler constant i.e. 2.71828
X=[2.3,4.5,-0.9]
exp_value=[]
normalized_value=[]
for value in X:
    exp_value.append(E**value)
    sum=0+value
for final in exp_value:
    normalized_value.append(final/sum)

print(normalized_value)    



