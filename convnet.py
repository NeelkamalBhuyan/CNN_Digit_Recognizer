
import numpy as np

from all_layers import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of shape (N, C, H, W)
    
    """

    def __init__(self,input_dim,num_filters,filter_size,hidden_dim,num_classes,weight_scale=1e-3,reg=0.0,dtype=np.float32,):
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # padding and stride of the first convolutional layer are chosen so that
        # the width and height of the input are preserved
        # random initialisation of params with gaussian distribution (weight_scale is its std deviation)
        C, H, W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale*np.random.randn(num_filters*H*W//4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale*np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        
        # forward pass through the network
        c1, cache_conv = conv_forward_naive(X, W1, b1, conv_param)
        r1, cache_relu1 = relu_forward(c1)
        p1, cache_pool = max_pool_forward_naive(r1, pool_param)
        fc1, cache_fc1 = affine_forward(p1, W2, b2)
        r2, cache_relu2 = relu_forward(fc1)
        fc2, cache_fc2 = affine_forward(r2, W3, b3)
        probs = np.exp(fc2)/np.sum(np.exp(fc2), axis=1, keepdims=True)
        scores = fc2
        

        if y is None:    # implies test time
            return scores

        loss, grads = 0, {}    # for training time
        
        num_examples = X.shape[0]
        # softmax loss function
        loss = (((-1)/num_examples)*np.sum(np.log(probs[np.arange(num_examples),y]))) + 0.5*self.reg*(np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2'])+ np.sum(self.params['W3']*self.params['W3']))
        # now back prop
        dZ3 = probs
        dZ3[np.arange(num_examples),y] -= 1
        dZ3 = dZ3/num_examples                          # gradient wrt scores
        dA2, dW3, db3 = affine_backward(dZ3, cache_fc2)
        dZ2 = relu_backward(dA2, cache_relu2)
        dP1, dW2, db2 = affine_backward(dZ2, cache_fc1)
        dA1 = max_pool_backward_naive(dP1, cache_pool)
        dZ1 = relu_backward(dA1, cache_relu1)
        dA0, dW1, db1 = conv_backward_naive(dZ1, cache_conv)
        grads['W3'] = dW3 + self.reg*W3
        grads['W2'] = dW2 + self.reg*W2
        grads['W1'] = dW1 + self.reg*W1
        grads['b3'] = db3
        grads['b2'] = db2
        grads['b1'] = db1
        

        

        return loss, grads
