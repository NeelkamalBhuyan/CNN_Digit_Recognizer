
import numpy as np



def affine_forward(x, w, b):
    """
    Computes forward pass for an affine (fully-connected) layer.

    """
    
    num_examples = x.shape[0]
    x_flat = x.reshape(num_examples,-1)
    out = np.dot(x_flat,w) + b.reshape(b.shape[0],1).T
    

    
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes backward pass for an affine layer.

    """
    x, w, b = cache
    
    num_examples = x.shape[0]
    x_flat = x.reshape(num_examples,-1)
    dw = np.dot(x_flat.T,dout)
    db = np.sum(dout,axis=0)
    dx = np.dot(dout,w.T)
    dx = dx.reshape(x.shape)
    
    
    return dx, dw, db


def relu_forward(x):
    """
    Computes forward pass for a layer of rectified linear units (ReLUs).

    """
    
    out = np.maximum(x,0)
    

    
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes backward pass for a layer of rectified linear units (ReLUs).

    """
    x = cache
    
    derivative_x = x>0
    dx = dout*derivative_x
    

    
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    forward pass for a convolutional layer.

    """
    
    pad = conv_param['pad']
    stride = conv_param['stride']
    N = x.shape[0]
    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[3]
    H = x.shape[2]
    W = x.shape[3]
    H1 = int(1 + (H + 2 * pad - HH) / stride)       # output height
    W1 = int(1 + (W + 2 * pad - WW) / stride)       # output width
    out = np.zeros((N, F, H1, W1))
    x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    for f in range(F):
        filt = w[f,:,:,:]
        row = 0
        col = 0
        out_r=0
        out_c=0
        while(row < x_padded.shape[2]):             # convolution loop
            if row+HH-1 > x_padded.shape[2]-1: break
            while(col < x_padded.shape[3]):
                if col+WW-1 > x_padded.shape[3]-1: break
                conv = np.sum(filt*x_padded[:,:,row:row+HH,col:col+WW],axis=(1,2,3)) + b[f]     # convolution
                out[:,f,out_r,out_c] += conv
                col += stride
                out_c +=1
            row += stride
            out_r += 1
            col = 0
            out_c = 0
    

   
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    backward pass for a convolutional layer.

    """
    
    x, w, b, conv_param = cache
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    dx = np.zeros(x.shape)
    pad = conv_param['pad']
    stride = conv_param['stride']
    N = x.shape[0]
    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[3]
    H = x.shape[2]
    W = x.shape[3]
    H1 = int(1 + (H + 2 * pad - HH) / stride)
    W1 = int(1 + (W + 2 * pad - WW) / stride)
    dx_padded = np.pad(dx, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    for f in range(F):
        filt = w[f,:,:,:]
        for i in range(N):
            img = x_padded[i,:,:,:]
            row = 0
            col = 0
            out_r=0
            out_c=0
            while(row < x_padded.shape[2]):             # convolution loop
                if row+HH-1 > x_padded.shape[2]-1: break
                while(col < x_padded.shape[3]):
                    if col+WW-1 > x_padded.shape[3]-1: break
#                     conv = np.sum(filt*img[:,row:row+HH,col:col+WW]) + b[f]
#                     out[i,f,out_r,out_c] += conv
                    dw[f,:,:,:] += img[:,row:row+HH,col:col+WW]*dout[i,f,out_r,out_c]        # convolution gradient wrt params
                    dx_padded[i,:,row:row+HH,col:col+WW] += w[f,:,:,:]*dout[i,f,out_r,out_c] # convolution gradient wrt layer input
                    db[f] += dout[i,f,out_r,out_c]
                    col += stride
                    out_c += 1
                row += stride
                out_r += 1
                col = 0
                out_c = 0
    dx = dx_padded[:,:,pad:pad+H,pad:pad+W]     # discarding padded part as its not part of input
    

    
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    forward pass for a max-pooling layer.

    """
    
    N = x.shape[0]
    stride = pool_param['stride']
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    H = x.shape[2]
    W = x.shape[3]
    H1 = int(1 + (H - HH) / stride)
    W1 = int(1 + (W - WW) / stride)
    out = np.zeros((N, x.shape[1], H1, W1))
    row = 0
    col = 0
    out_r=0
    out_c=0
    while(True):
        if row+HH-1 > H-1: break
        while(col < W):
            if col+WW-1 > W-1: break
            out[:,:,out_r,out_c] = np.amax(x[:,:,row:row+HH,col:col+WW],axis=(2,3)) #calculaing max only on 2D part
            col += stride
            out_c +=1
        row += stride
        out_r += 1
        col = 0
        out_c = 0
    

    
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    backward pass for a max-pooling layer.

    """
    
    (x, pool_param) = cache
    dx = np.zeros(x.shape)
    N = x.shape[0]
    stride = pool_param['stride']
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    H = x.shape[2]
    W = x.shape[3]
    for i in range(N):
        row = 0
        col = 0
        out_r=0
        out_c=0
        for d in range(x.shape[1]):
            row = 0
            col = 0
            out_r=0
            out_c=0
            while(True):                    # moving across 2D part
                if row+HH-1 > H-1: break
                while(True):
                    if col+WW-1 > W-1: break
                    num = np.argmax(x[i,d,row:row+HH,col:col+WW]) # finding the value that contributed to pool output
                    r = num//WW
                    c = num%WW
                    dx[i, d, row+r, col+c] = dout[i, d, out_r, out_c] # only that value will have non-zero gradient
                    col += stride
                    out_c +=1
                row += stride
                out_r += 1
                col = 0
                out_c = 0
    

    
    return dx

