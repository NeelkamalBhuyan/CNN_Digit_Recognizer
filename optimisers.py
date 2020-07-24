import numpy as np


def optimiser_type(optim_type, w, dw, config=None):
    
    if optim_type=="sgd":
        
        if config is None:
            config = {}
        config.setdefault("learning_rate", 1e-2)
    
        w -= config["learning_rate"] * dw

    
    elif optim_type=="sgd_momentum":

        if config is None:
            config = {}
        config.setdefault("learning_rate", 1e-2)
        config.setdefault("momentum", 0.9)
        v = config.get("velocity", np.zeros_like(w))

        next_w = None
        
        v = config["momentum"]*v - config["learning_rate"]*dw # calculating new velocity (or first moment of grads)
        next_w = w + v                                          # param update
        
        config["velocity"] = v                                  # velocity update
    

    
    elif optim_type=="rmsprop":
        
        if config is None:
            config = {}
        config.setdefault("learning_rate", 1e-2)
        config.setdefault("decay_rate", 0.99)
        config.setdefault("epsilon", 1e-8)
        config.setdefault("cache", np.zeros_like(w))        # cache holding second moment of gradients
    
        next_w = None
        
        config["cache"] = config["decay_rate"]*config["cache"] + (1-config["decay_rate"])*dw**2 # updating cache with new second moment
        next_w = w - config["learning_rate"]*(dw/(np.sqrt(config["cache"]) + config["epsilon"])) # update
        

    
    elif optim_type=="adam":

        if config is None:
            config = {}
        config.setdefault("learning_rate", 1e-3)
        config.setdefault("beta1", 0.9)
        config.setdefault("beta2", 0.999)
        config.setdefault("epsilon", 1e-8)
        config.setdefault("m", np.zeros_like(w))            # first moment of grads
        config.setdefault("v", np.zeros_like(w))            # second moment of grads
        config.setdefault("t", 0)                           # tracking iteration number for normalising the moment
    
        next_w = None
        
        config["t"]=config["t"]+1
        config["m"] = config["beta1"]*config["m"] + (1-config["beta1"])*dw # update to first moment
        mt = config["m"]/(1-config["beta1"]**config["t"])                   # normalising
        config["v"] = config["beta2"]*config["v"] + (1-config["beta2"])*(dw**2) # update to second moment
        vt = config["v"]/(1-config["beta2"]**config["t"])                   # normalising
        next_w = w - config["learning_rate"]*mt/(np.sqrt(vt) + config["epsilon"]) # update to params
        
        
    return next_w, config
