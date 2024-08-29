import numpy as np
import neurolab as nl
from prettytable import PrettyTable
from datetime import datetime as dt

_NEURONS_= 5        #Neurons number in the hidden layer
_LOW_= -0.5         #Low limit for data generation
_HIGH_= 0.5         #High limit for data generation
_SIZE_= 10          #Size of training set to the net
_FILE_NAME_= f'test_{dt.today().strftime("%d-%m-%Y %H-%M-%S")}.net'

#Create train samples (a matrix _SIZE_x2 size with a normal distribution between -0.5 and 0.5) 
X= np.random.uniform(_LOW_, _HIGH_, (_SIZE_, 2))
target= (X[:, 0] + X[:, 1]).reshape(_SIZE_, 1)

#Create the network: 2 inputs, _NEURONS_ neurons in input layer and 1 output layer
net= nl.net.newff([[_LOW_, _HIGH_], [_LOW_, _HIGH_]], [_NEURONS_, 1])

#Train the network
err= net.train(X, target)

#Test the network
test= np.random.uniform(_LOW_, _HIGH_, (3, 2))
Y = net.sim(test)

#Show results
table= PrettyTable()
table.align = "l"
table.field_names=['x1', 'x2', 'y', 'Expec', 'RelErr']
for i in range(Y.__len__()):
    x1= f'{X[i][0]:.3f}'
    x2= f'{X[i][1]:.3f}'
    y= f'{Y[i][0]:.3f}'
    expec= f'{(X[i][0] + X[i][1]):.3f}'
    relerr= f'{(np.absolute((float(expec) - float(y))/float(expec))*100):.0f}%'
    table.add_row([x1, x2, y, expec, relerr])
print(table.get_string() + '\n')

save= input('Do you want to save the network? y/n\n')
if('y' in save.lower()):
    net.save(_FILE_NAME_)