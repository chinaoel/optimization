import random
from engine import MLP

XOR_X = [[1,1],
         [1,0],
         [0,1],
         [0,0]]

XOR_Y = [1,
         -1,
         -1,
         1]

random.seed(1028)



mlp = MLP((2,4,1))



epoch = 10



for i in range(epoch):

    # zero grad

    for params in (mlp.parameters()):

        params.grad = 0



    outputs = []

    # forward pass

    for each in XOR_X:

        output = (mlp(each))

        outputs.append(output) 

    

    # calculate gradient from loss

    loss = (sum([(y - y_bar)**2 for y_bar,y in zip(XOR_Y,outputs)]))

    loss.backward()



    for params in (mlp.parameters()):

        params.data += (-0.01) * params.grad



    

    print(f"Epoch : {i}, Loss: {loss}")



print(outputs)