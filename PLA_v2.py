# -*- coding: utf-8 -*-
"""
Created on Fri April 08 10:38:33 2015

@author: LiangLeon
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def dot_product(values, weights):
    "Do some dot production"
    return sum(value * weight for value, weight in zip(values, weights))

def separable_2d(seed, n_points, classifier):
    np.random.seed(seed)
 
    dim_x = 2
    data_dim = dim_x + 1 # leading 1 and class value
    data = np.ones(data_dim * n_points).reshape(n_points, data_dim)
 
    # fill in random values
    data[:, 0] = -1 + 2*np.random.rand(n_points)
    data[:, 1] = -1 + 2*np.random.rand(n_points)
 
    # TODO: use numpy way of applying a function to rows.
    for idx in range(n_points):
        if classifier.class_of(data[idx]) == 1:
            data[idx,-1] = 1
        else:
            data[idx,-1] = 0          
 
    return data

if __name__ == '__main__':
    #Training parameters
    threshold = 1      #Bias
    learning_rate = 0.5  #Learning rate
    weights = [0, 0]   #Initial weights
    #Prepare you training set here
    import linear_classifier
    data_dim = 2
    classifier = linear_classifier.Classifier()
    classifier.init_random_last0(data_dim + 1, 130216)
    
    data = separable_2d(263245, 16, classifier)
 
    condition = data[:, 2] == 1
    positive = np.compress(condition, data, axis=0)
 
    neg_condition = data[:, 2] == 0
    negative = np.compress(neg_condition, data, axis=0)
    
    training_set = []
    class1_sample = positive[:,0:2].T
    class2_sample = negative[:,0:2].T
    for row in data:
        training_set.append( (tuple(row[0:2]), row[2]) )
    line_function = []

    #PLA main body
    while True:
        error_count = 0
        for input_vector, desired_output in training_set:
            result = dot_product(input_vector, weights) > threshold
            error = desired_output - result
            if error != 0:
                error_count += 1
                #store line function for plot
                linef = weights +  [ -1 * threshold ]
                line_function.append( linef )
                #update weights
                for index, value in enumerate(input_vector):
                    weights[index] += learning_rate * error * value
        if error_count == 0:
            break
    #store last line function for plot
    linef = weights +  [ -1 * threshold ]
    line_function.append( linef )
    
    
    #Plot data
    StartPoint = -2
    EndPoint = 2
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(xlim=(-2, 2), ylim=( -2, 2))
    line, = ax.plot([], [], lw=2)
    ax.plot( class1_sample[0,:], class1_sample[1,:], 'o', markersize=10, color='blue', alpha=0.5, label='class1')
    ax.plot( class2_sample[0,:], class2_sample[1,:], '^', markersize=10, alpha=0.5, color='red', label='class2' )
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        x = [StartPoint, EndPoint]
        if line_function[i][1] != 0:
            y = [ ( threshold - line_function[i][0] * StartPoint )  /line_function[i][1], ( threshold - line_function[i][0] * EndPoint  ) / line_function[i][1]   ]
        else:
            y = [ threshold - line_function[i][0] * StartPoint, threshold - line_function[i][0] * EndPoint ]
        line.set_data(x, y)
        return line,    
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(line_function), interval=1000, blit=True, repeat=False)
    plt.show()