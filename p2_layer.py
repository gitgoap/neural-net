input=[1,2,3,4]
weight1=[1,20,-30,6.0]
weight2=[2,4,-5.6,1]
weight3=[3,24,5,-1]
bias1=-20
bias2=3
bias3=3.0
output=[input[0]*weight1[0]+input[1]*weight1[1]+input[2]*weight1[2]+input[3]*weight1[3]+bias1,
       input[0]*weight2[0]+input[1]*weight2[1]+input[2]*weight2[2]+input[3]*weight2[3]+bias2,
       input[0]*weight3[0]+input[1]*weight3[1]+input[2]*weight3[2]+input[3]*weight3[3]+ bias3]
print(output)

'''
3 neuron having common 4 input with 3 unique bias
'''