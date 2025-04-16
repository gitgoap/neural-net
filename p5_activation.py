from nnfs.datasets import spiral_data
import numpy as np
np.random.seed(0)

#nnfs.init()

X,y =spiral_data(100,3)

class dense_layer:
    def __init__(self,n_layer_neuron, next_layer_neuron):
        self.weight=np.random.randn(n_layer_neuron, next_layer_neuron)
        self.bias = np.zeros((1,next_layer_neuron))
    def feed_forward(self, input):
        self.output=np.dot(input,self.weight)+self.bias
class activation:
    def forward(self,prefinal):
        self.output=np.maximum(0,prefinal)

layer1=dense_layer(2,5)
layer1a=activation()

layer1.feed_forward(X)
layer1a.forward(layer1.output)
 

print(layer1a.output)
           
