# CNN_Digit_Recognizer

Using Convolutional Neural Network for Digit Recognizer

This project is a submission for the Kaggle Digit Recognizer Challenge

The following approach has been followed:
  1. Python with Numpy have been used
  2. Neural Network Outline: conv -> relu -> 2x2 max pool -> affine -> relu -> affine -> softmax
  3. A class, ThreeLayerConvNet, in convnet.py, has been created for the neural net having:
  
      a. Parameters in a dictionary
      
      b. A constructor initialising those parameters with gaussian random variable of a given standard deviation
      
      c. function calculating the loss and gradients
      
  4. A module, all_layers.py, having functions for forward and backward pass for different types of layers used.
  5. A module, optimisers.py, performing parameter update based on type of optimiser chosen (all first order)
  6. A class, Solver, in solver_trainer.py, which takes in the model, training data and choice of hyperparameters and performs mini-batch stochastic gradient descent for given number of epochs
  7. Forward and Backward pass for all layers have been written in Numpy
  8. Parameter updates for all optimisers have been written in Numpy
  9. Descriptions for all files, along with the functions in them, have been provided for explaining the operations performed in them
  10. Most functions have the option of using appropriate default values (mainly in case of hyperparameters) when they are not provided (BONUS)
  11. main.py program involves:
  
       a. Importing training and test data and splitting training data into training and validation
       
       b. Data preprocessing for faster convergence (centering the data)
       
       c. Input data has images in form of 1D arrays, so they have been reshaped to (28,28) size of depth=1
       
       d. Training the model
       
       e. Calculating predictions for test data
       
       f. Converting predictions to .csv file of required format
