# Minimal neural network demo for Octave (Matlab)

Based on equations from the first chapters of [M. Nielsen's book](http://neuralnetworksanddeeplearning.com/index.html) 
Trains a NN for digit recognition on a set of 5000 20x20 images from the NIST dataset.

* 400 nodes input layer (20x20 pixel)
* 30 hidden nodes
* 10 output nodes (digits 0..9)

Recognition of the training dataset converges to 99.0 % after a minute or two.