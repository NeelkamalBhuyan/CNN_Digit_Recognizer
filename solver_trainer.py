
import numpy as np

from optimisers import *


class Solver(object):
    """
    The Solver performs stochastic gradient descent using different
    update rules defined in optimisers.py.

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data.

    """

    def __init__(self, model, data, batch_size=50, num_epochs=2, optim_type="adam", optim_config={'learning_rate': 1e-2,}, lr_decay=1.0, num_train_samples=100, num_val_samples=None, verbose=True):
        """
        Construct a new Solver instance.

        """
        self.model = model
        
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # Setting up variables for the hyperparameters
        
        self.optim_type = optim_type
        self.optim_config = optim_config    # dict containing hyperparameters related to parameter update
        self.lr_decay = lr_decay            # learning rate decay rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples

        self.print_every = 20
        self.verbose = verbose
        
        # Setting up some extra variables for faster convergence / book-keeping
        
        self.epoch = 0            # to keep track of number of epochs done
        self.best_val_acc = 0     # to keep track of the best val accuracy across all epochs
        self.best_params = {}     # to keep track of best model across all epochs
        self.latest_loss = 0      # to keep track of loss in latest iteration

        # Making a copy of the optim_config for each parameter
        # for using in other functions of the solver class
        # optim_cofig contains first and second moment of gradients, if applicable, wrt 1 param and hence each parameter has its own optim_config dict
        
        self.optim_configs = {}         # dictionary containing config dicts of all params
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}    # copying the input config dict to config dicts of all params
            self.optim_configs[p] = d
 

    def single_step(self):
        """
        Making a single gradient update. This is called by training()
        """
        # Make a minibatch of training data by choosing "batch_size" elements with replacement
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size) # random choice with replacement
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.latest_loss = loss

        # Perform a parameter update based on chosen optimiser
        for p, w in self.model.params.items():
            dw = grads[p]                       # current gradients
            config = self.optim_configs[p]      # moments of gradients and learning rate till previous accuracy() call
            next_w, next_config = optimiser_type(self.optim_type, w, dw, config)     # sent to choice of optimising technique
            self.model.params[p] = next_w       # model params updated
            self.optim_configs[p] = next_config # # moments of gradients updated
        

    def accuracy(self, X, y, num_samples=None, batch_size=100):
        
        # subsampling the data to the number of examples (less than total) to be used to check accuracy
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)     # random choice with replacement
            N = num_samples
            X = X[mask]
            y = y[mask]

        scores = self.model.loss(X) # not sending y returns only scores from loss function
        y_pred = np.argmax(scores,axis=1)   # input classified into class with max score
        if y is None:               # using this as a test case (no label) prediction function at the same time
            return y_pred
        acc = np.mean(y_pred == y)  # function will return if input is training example (y in not None)

        return acc

    def training(self):
        """
        Running optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch     # total number of iterations

        for t in range(num_iterations):
            self.single_step()              # single step of param update with a minibatch

            # printing training loss only after every "print every" number of iterations
            if self.verbose and t % self.print_every == 0:
                print("(Iteration %d / %d) loss: %f" % (t + 1, num_iterations, self.latest_loss))

            # At the end of every epoch, incrementing the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0 # boolean indicating whether one epoch is done
            if epoch_end:
                self.epoch += 1                             # epoch increment
                for k in self.optim_configs:                # updating learning rate of all params
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay # learning rate decayed after each epoch

            # Checking train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
                    
            if t == 0 or t == (num_iterations - 1) or epoch_end:
                # checking train accuracy
                train_acc = self.accuracy(self.X_train, self.y_train, num_samples=self.num_train_samples)
                # checking val accuracy
                val_acc = self.accuracy(self.X_val, self.y_val, num_samples=self.num_val_samples)

                if self.verbose:               # printing train_acc and val_acc every epoch
                    print("(Epoch %d / %d) train acc: %f; val_acc: %f" % (self.epoch, self.num_epochs, train_acc, val_acc))

                # Keeping track of the best model at every epoch (not iteration)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
                        
        # self.model.params gets updated every iteration. This leads to noise created
        # created by minibatch sgd
        # Hence at the end of an epoch swapping the best params into the model
        # for faster optimisation
        self.model.params = self.best_params
