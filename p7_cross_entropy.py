# cross-entropy for classification problem
import numpy as np
predicted_output=[0.7, 0.1, 0.2]  # These output came from softmax function called confidence value
target_output=   [1, 0, 0]        #0.7 should be 1 others should be 0, like a one-hot vector

cross_entropy_loss = -(np.log(predicted_output[0])*target_output[0] + 
                       np.log(predicted_output[1])*target_output[1] +
                       np.log(predicted_output[2])*target_output[2])

print(cross_entropy_loss)

# verification
# import math
# print(math.e **- 0.35667494393873245)