import numpy as np,jax,jax.numpy as jnp,json,abc,os,string,random
from file_manager import store_file,load_file,access_locked_file,MODEL_DIRECTORY
FLOAT_T= jnp.float32
MAX_DATA_SIZE = 30000
# MAX_DATA_SIZE = 2500


def softmax(y,axis=-1):
    fs = jnp.exp(y - jnp.amax(y,axis=axis,keepdims=True))
    return fs / jnp.sum(fs,axis=axis,keepdims=True)

ACTIVATION = {
    "relu":lambda s:jnp.where(s < 0,0,s),
    "sigmoid":lambda s,temp=1:jax.nn.sigmoid(s*temp),#1/(1+jnp.exp(-1 * s * temp))
    "tanh":jnp.tanh,
    "softmax":jax.nn.softmax,
    "prelu":lambda s,p=4:jnp.where(s < 0,s*p,s),
    "softplus":lambda s,b=1:jax.nn.softplus(s*b)/b,#jnp.log(1 + jnp.exp(s * b))/b
    "silu":jax.nn.silu #x / (1+jnp.exp(-s))
}

def get_activation_f(name):
    if not "_" in name:
        return ACTIVATION[name]
    args = name.split("_")
    return lambda s:ACTIVATION[args[0]](s,*[float(a) for a in args[1:]])  
      

def lookup_layer(Ws,X):
    if len(Ws) > 2:
        Z=0
        for i in range(len(Ws)):
            Z += jnp.take(Ws[i],X[:,:,-(len(Ws)-i)].astype(jnp.int32),axis=0,mode="clip")
        return Z
    elif len(Ws) == 1:
        return jnp.take(Ws[0],X[:,:,-1],axis=0)
    return 0

def embedding_layer(Ws,X,pos_enc=None):
    W_p,W_lookup = Ws
    Z = 0 if W_p.shape[0] == 0 else X[:,:,:W_p.shape[0]] @ W_p
    Z += lookup_layer(W_lookup,X[:,:,W_p.shape[0]:])
    # Z = lookup_layer(W_lookup,X) if W_p.shape[0] == 0 else X[:,:,:W_p.shape[0]] @ W_p
    if pos_enc is not None:
        return Z + pos_enc[:Z.shape[1],:]
    return Z

linear_layer = lambda W,X: X @ W[1:, :] + W[0:1, :]

dropout_layer = lambda X,dropout: X if dropout is None else (X * jax.random.binomial(dropout[0],1,1-dropout[1],shape=(1,X.shape[-1])))

normalize = lambda X,W=1,b=0,axis=-1: ( ( (X - X.mean(axis, keepdims=True))  * W)  / X.std(axis, keepdims=True)) +b

RMSnorm = lambda X,W=1: X * jax.lax.rsqrt(jnp.mean(X**2,axis=-1,keepdims=True) + 1e-12) * W


def cdma_decode(Z,gates):
    if gates is None:
        return Z
    return (Z.swapaxes(1,2) @ gates).swapaxes(1,2)


def cdma_encode(Ws,X):
    if len(Ws) == 1:
        return X @ Ws[0],None
    feature_Ws,gate_Ws = Ws
    features = linear_layer(
        feature_Ws,X
    ).swapaxes(1,2)
    gates = softmax(
        linear_layer(gate_Ws,X).swapaxes(1,2)/jnp.sqrt(gate_Ws.shape[-1])
    )
    Z = (features @ gates.swapaxes(1,2)).swapaxes(1,2)
    return Z,gates

def fc_layer(Ws,act_func,X,dropout=None):
    for i in range(len(Ws)-1):
        X = dropout_layer(act_func(linear_layer(Ws[i],X)),dropout=dropout)
    return linear_layer(Ws[-1],X)

def prune_layer(W_fc,W_mask,act_func,X):
    X = act_func(linear_layer(W_fc,X))
    return softmax(W_mask) * X


class Pruning:
    def prune_Ws(weights,scores,k=0.5):
        index = scores.flatten().argsort()
        return weights.flatten().at[index[:int((1-k)*index.shape[0])]].set(0).reshape(weights.shape)

    @jax.custom_vjp
    def prune(weights,scores,k=0.5):
        return Pruning.prune_Ws(weights,scores,k)

    def prune_fwd(weights,scores,k=0.5):
        return Pruning.prune(weights,scores,k=k),jnp.abs(weights)

    def prune_bwd(res,grad):
        return (jnp.ones(res.shape),grad*res)

    prune.defvjp(prune_fwd,prune_bwd)

    def apply_pruning(weights,scores=None,k=0.5):
        if scores is None:
            return weights
        if isinstance(weights, jax.numpy.ndarray):
            return Pruning.prune(weights,scores,k=k)
        return [Pruning.prune(weights[i],scores[i],k=k) for i in range(len(weights))]

class tools:
    LOSS_FUNCS = {
        "nll":lambda Y,T:-jnp.mean(T * jnp.log(Y+1e-12)),
        "mse":lambda Y,T:jnp.mean((T-Y)**2)
    }
    COMPILED = {}
    def flatten_grad(grad,size,argnums=[0]):
        def _flatten_grad(grad):
            if type(grad) == tuple:
                return [_flatten_grad(g) for g in grad]
            elif type(grad)==list:
                gs = []
                for g in grad:
                    flattened = _flatten_grad(g)
                    if flattened is not None:
                        gs.append(flattened)
                return jnp.hstack(gs,dtype=FLOAT_T)
            if grad.size == 0:
                return None
            return grad.flatten()
        return _flatten_grad(grad)/size if len(argnums) <= 1 else [_flatten_grad(grad[0][i])/size if i in argnums else None for i in range(max(argnums)+1)]
    
    def get_positional_encoding(model_dim,seq_length):
        denom = np.power(10,np.arange(model_dim)[None,:].astype(np.float32) / model_dim)
        radians = np.arange(seq_length)[:,None] / denom
        radians[:,0::2] = np.sin(radians[:,0::2])
        radians[:,1::2] = np.cos(radians[:,1::2])
        return jnp.array(radians,dtype=FLOAT_T)
    
    # def flatten_grad(grad,size,argnums=1):
    #     def _flatten_grad(grad):
    #         if type(grad) == tuple:
    #             return [_flatten_grad(g) for g in grad]
    #         elif type(grad)==list:
    #             gs = []
    #             for g in grad:
    #                 flattened = _flatten_grad(g)
    #                 if flattened is not None:
    #                     gs.append(flattened)
    #             return jnp.hstack(gs,dtype=FLOAT_T)
    #         if grad.size == 0:
    #             return None
    #         return grad.flatten()
    #     return _flatten_grad(grad)/size if argnums <= 1 else [_flatten_grad(grad[0][i])/size for i in range(argnums)]
    
    # def compile_model(loss,forward_f,argnums=1,name=None,dropout=0.0,**kwargs):
    def compile_model(loss,forward_f,argnums=[0],name=None,**kwargs):
        loss_f = tools.LOSS_FUNCS[loss] if type(loss) is str else loss
        if (name is None) or (not name in tools.COMPILED):
            forward = jax.jit(forward_f)
            error_f = lambda Ws,X,T,*args,**kwargs:loss_f(forward(Ws,X,*args,**kwargs),T)
            grad_f = jax.jit(jax.grad(error_f,argnums=[0]))      
            # if dropout > 0:
            #     rng_key = tools.rng_key()
            compiled = {
                "error":jax.jit(error_f),
                #"grad":lambda W,X,T,*args,**kwargs:tools.flatten_grad(grad_f(W,X,T,*args,**(kwargs if dropout <= 0 else {"dropout":(next(rng_key),dropout),**kwargs})),np.product(X.shape[:-1])*T.shape[-1],argnums=argnums),
                "grad":lambda W,X,T,*args,**kwargs:tools.flatten_grad(grad_f(W,X,T,*args,**kwargs),np.product(X.shape[:-1])*T.shape[-1],argnums=argnums),
                "forward":forward
            }
            if name is not None:
                tools.COMPILED[name] = compiled
            return compiled
        return tools.COMPILED[name]
    
    def get_fc_shapes(n_inputs,n_outputs,n_hiddens=[80],**kwargs):
        layers = [n_inputs] + n_hiddens + [n_outputs]
        return [(layers[i]+1,layers[i+1]) for i in range(len(layers)-1)]

    def make_weights(shape):
        if len(shape)==0:
            shape = [0]
        if type(shape[0]) != int:
            return np.hstack([tools.make_weights(s) for s in shape])
        return np.random.uniform(-1,1,size=(np.prod(shape),)).flatten() / np.sqrt(shape[0])
        
    def reshape_weights(_all_weights,_shapes):
        def _reshape_weights(all_weights,shapes):
            if len(shapes)==0:
                return jnp.zeros((0)),0
            if type(shapes[0]) == int:
                size = np.product(shapes)
                return jnp.array(all_weights[:size].reshape(shapes),dtype=FLOAT_T),size
            start = 0
            Ws = []
            for shape in shapes:
                r_ws,size = _reshape_weights(all_weights[start:],shape)
                Ws.append(r_ws)
                start += size
            return Ws,start
        return _reshape_weights(_all_weights,_shapes)[0]


    def rng_key(random_state = None):
        key = jax.random.key(np.random.randint(0,99999999) if random_state is None else random_state)
        while True:
            key,sk = jax.random.split(key)
            yield sk

class Optimizers:

    def adamw_step(grad,W,state,params,learning_rate):
        ##(mt,vt,beta1t,beta2t),(beta1,beta2,w_decay,learning_rate)
        beta1,beta2,w_decay = params
        mt,vt,beta1t,beta2t = state
        mt = beta1 * mt + (1 - beta1) * grad
        vt = beta2 * vt + (1 - beta2) * grad * grad
        beta1t = beta1 * beta1t
        beta2t = beta2 * beta2t
        m_hat = mt / (1 - beta1t)
        v_hat = vt / (1 - beta2t)
        W_n = W - ((learning_rate * m_hat / (jnp.sqrt(v_hat) + 1e-16)) + (W * w_decay))
        return W_n,(mt,vt,beta1t,beta2t)

    OPTIMIZER_FUNCS = {
        "adamw":(lambda W_shape,beta1=0.9,beta2=0.999,w_decay=0.0,**kwargs:((np.zeros(W_shape),np.zeros(W_shape),1,1),(beta1,beta2,w_decay)),adamw_step),
        "SGD":(lambda W_shape,**kwargs:(None),lambda grad,W,state,learning_rate:(W-(grad*learning_rate),None))
    }
    OPTIMIZERS = {}

    def get_ln_normalizer(reg_constant,n):
        if n<1 or reg_constant < 1:
            return lambda W:0
        return lambda W: (jnp.abs(W) **(n-1)) * (reg_constant)

    class Optimizer:
        def __init__(self,W_shape,optimizer="adamw",learning_rate=0.01,ln_norm=2,reg_constant=0,**kwargs):
            funcs = Optimizers.OPTIMIZER_FUNCS[optimizer]
            if not optimizer in Optimizers.OPTIMIZERS:
                Optimizers.OPTIMIZERS[optimizer] = jax.jit(funcs[1])
            self.state,self.params = funcs[0](W_shape,**kwargs)
            self.learning_rate = learning_rate
            self.step_f = Optimizers.OPTIMIZERS[optimizer]
            self.ln_normalizer = Optimizers.get_ln_normalizer(reg_constant,ln_norm)

        def step(self,grad,W):
            if grad is None:
                return W
            grad += self.ln_normalizer(W)
            w_n,self.state = self.step_f(grad,W,self.state,self.params,self.learning_rate)
            return np.array(w_n)
        
        def adapt_learning_rate(self,lr_decay_factor=3,lr_early_exit=10e-5,**kwargs):
            self.learning_rate = self.learning_rate / lr_decay_factor
            return self.learning_rate > lr_early_exit


class Options:
    class RandomGen:
        def __init__(self,p=0.1,all_axes=False):
            self.rng_key = tools.rng_key()
            self.dropout = p
        def __call__(self,inference=False):
            return (next(self.rng_key),self.dropout)

    class Dropout(RandomGen):
        def __init__(self,p=0.1,all_axes=False):
            super().__init__(p=p)
            self.get_shape = jax.tree_util.Partial((lambda shape:shape) if all_axes else (lambda shape:(1,shape[-1])))
        def __call__(self,inference=False):
            return None if (inference or self.dropout<= 0) else (next(self.rng_key),self.dropout,self.get_shape)
        
    # class Observer:
    #     pass

    #     ## use for collecting layer-norm statistics to return during training
    #     def __call__(self,inference=False):
    #         ## return statistic if inference is False and save in moving average
    #         ## else return moving average
    #         return None


class NNets:
    class Parameters:
        class Weights:
            def __init__(self,shapes,weights=None):
                if weights is None:
                    weights = tools.make_weights(shapes)
                    weights[np.where(weights)==0] = 1e-8
                self.weights = np.array(weights)
                self.shapes = shapes
                self.shape = self.weights.shape
            def get_weights(self):
                return tools.reshape_weights(self.weights,self.shapes)
            def as_dict(self):
                return {"shapes":self.shapes,"weights":self.weights.tolist()}
            def prune(self,previous_weights=0,prune_rate=0.2):
                i = np.argsort(np.absolute(self.weights-previous_weights))[:int(self.weights.shape[0] * prune_rate)]
                self.weights[i] = 0
            
        
        def fully_connected(n_inputs,n_outputs,**kwargs):
            return NNets.Parameters(tools.get_fc_shapes(n_inputs,n_outputs,**kwargs),**kwargs)
        
        def embedding(data_model,model_dim=40,**kwargs):
            return NNets.Parameters(( 
                (data_model.n_inputs,model_dim), [(nl+1,model_dim) for nl in data_model.n_lookups] 
            ))
        
        def __init__(self,shapes=(),weights=None,path=None,n_outputs=None,requires_grad=True,**kwargs):
            self.weights = NNets.Parameters.Weights(shapes,weights)
            self.path = path
            self.optimizer = Optimizers.Optimizer(self.weights.shape,**kwargs)
            self.n_outputs = n_outputs
            self.requires_grad = requires_grad
        
        def output_dim(self):
            return self.weights.shapes[-1][-1] if self.n_outputs is None else self.n_outputs

        def get_weights(self):
            return self.weights.get_weights()
        
        def set_coef(self,weights):
            self.weights.weights[:] = weights
        
        def get_coef(self):
            return self.weights.weights

        def step(self,grad):
            self.set_coef(self.optimizer.step(grad,self.get_coef()))
        
        def _store(self):
            return {"type":"parameters",**self.weights.as_dict(),"n_outputs":self.n_outputs}

        def __str__(self):
            return str(self.weights.shapes)
            # return str(self.weights.shape)
        
        def __repr__(self):
            def to_str(shape,t=""):
                if len(shape) == 0:
                    return t+"0"
                if type(shape[0]) is int:
                    return t+str(shape)
                return t+"[\n"+"\n".join([to_str(s,t+" ") for s in shape])+"\n"+t+"]"
            return to_str(self.weights.shapes)
        
        def size(self):
            return np.prod(self.weights.shape)
        
        def footprint(self):
            return int(self.weights.weights.nbytes)

    class ParameterContainer:
        def __init__(self,path=None,n_outputs=None,requires_grad=True,**params):
            self.path = path
            self.n_outputs = n_outputs
            self.requires_grad = requires_grad
            self.param_dict = params
            self.param_list = list(self.param_dict.values())

        def output_dim(self):
            return self.param_list[-1].output_dim() if self.n_outputs is None else self.n_outputs

        def get_weights(self):
            return [w.get_weights() for w in self.param_list]
        
        def _iter_vector(self):
            start = 0
            for p in self.param_list:
                end = start + p.size()
                yield (start,end),p
                start = end
        
        def set_coef(self,weights):
            [p.set_coef(weights[start:end]) for (start,end),p in self._iter_vector()]
            
        def get_coef(self):
            return np.hstack([p.get_coef() for p in self.param_list])

        def step(self,grad):
            [p.step(grad[start:end]) for (start,end),p in self._iter_vector()]

        def load(config):
            return NNets.ParameterContainer(n_outputs = config["n_outputs"],**{k:NNets.Parameters(**v) for k,v in config["parameters"].items()})

        def _store(self):
            return {"type":"container","parameters":{k:v._store() for k,v in self.param_dict.items()},"n_outputs":self.n_outputs}

        def __str__(self):
            return '\n'+'\n'.join(["\t"+k+ ": " + '\n\t'.join(str(v).split("\n")) for k,v in self.param_dict.items()])
            # return str({k:str(v) for k,v in self.param_dict.items()})
        
        def size(self):
            return int( np.sum([p.size() for p in self.param_list]) )
        
        def footprint(self):
            return int( np.sum([p.footprint() for p in self.param_list]) )

    class Module(abc.ABC):
        def _get_name(self,**kwargs):
            return self.model_name
        
        def __repr__(self):
            s = str(self.__class__)
            for k,v in self.parameters.items():
                s += f'\n  {k}: ({str(v)})'
            return s
        
        def __init__(self,data_model=None,max_data_size=MAX_DATA_SIZE,**kwargs):
            self.data_model = data_model
            self.max_data_size = max_data_size
            self.error_trace = []
            self.params = kwargs
            self.options = {}
            self.initialize(**kwargs)
            self.grad_batch_size = None
            self.error_batch_size = None
            self.forward_batch_size = None


        @abc.abstractmethod
        def initialize(self,**kwargs):
            pass

        def __setitem__(self,name,parameters):
            if not name in self.parameters:
                self.parameters[name] = parameters
            self.parameters[name].weights = parameters.weights.copy()

        def __getitem__(self,name):
            return self.parameters[name]
        
        def _get_weights(self,inference=False):
            Ws = [p.get_weights() for p in self.parameters.values()]
            return Ws if len(Ws)>1 else Ws[0]
        
        def _get_options(self,inference=False):
            if not hasattr(self,"options"):
                return {}
            return {k:v(inference=inference) if callable(v) else v for k,v in self.options.items()}
        
        def size(self):
            return int(np.sum([v.size() for v in self.parameters.values()]))
        
        def _error_f(self,*data):
            return self.compiled["error"](self._get_weights(),*data,**self._get_options(inference=True))

        def _grad_f(self,*data):
            return self.compiled["grad"](self._get_weights(),*data,**self._get_options(inference=False))
        
        def _forward_f(self,*data):
            return self.compiled["forward"](self._get_weights(inference=True),*data,**self._get_options(inference=True))
        
        def _run_dynamic_batch(self,f,*args,init_batch_size=None,min_batch_size=1):
            batch_size = int(init_batch_size or MAX_DATA_SIZE)
            size = args[0].shape[0] 
            while batch_size > min_batch_size:
                try:
                    if size <= batch_size:
                        return [(f(*args),1)],batch_size
                    return [(f(*[a[batch:batch+batch_size,:] if isinstance(a, jnp.ndarray) else a for a in args]),min(size-batch,batch_size)/size) for batch in range(0,size,batch_size)],batch_size
                except RuntimeError as e:
                    
                    if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
                        batch_size = batch_size // 2
                        print("\treducing batch size to",batch_size)
                    else: 
                        raise e
            raise Exception("unable to find batch size that fits memory")
        

        def grad_f(self,*args):
            res,self.grad_batch_size = self._run_dynamic_batch(self._grad_f,*args,init_batch_size=self.grad_batch_size)
            if type(res[0][0]) == list:
                grads = [(wg * res[0][1]) for wg in res[0][0]]
                for batch in res[1:]:
                    for i,g in enumerate(batch[0]):
                        grads[i] = grads[i] + (g * batch[1])
                return grads
            return np.sum([g[0] * g[1] for g in res],axis=0)
            # size = args[0].shape[0]
            # if size <= self.max_data_size:
            #     return self._grad_f(*args)
            # grads = None
            # is_list = False
            # for batch in range(0,size,self.max_data_size):
            #     multiplier = min(size-batch,self.max_data_size) / self.max_data_size
            #     grad = self._grad_f(*[a[batch:batch+self.max_data_size,:] if isinstance(a, jnp.ndarray) else a for a in args])
            #     if grads is None:
            #         is_list = (type(grad) == list)
            #         grads = [g*multiplier for g in grad] if is_list else grad*multiplier
            #         continue 
            #     if is_list:
            #         for i in range(len(grad)):
            #             grads[i] = grads[i] + (grad[i] * multiplier) 
            #     else:
            #         grads += (grad * multiplier)
            # return grads
        
        def error_f(self,*args):
            res,self.error_batch_size = self._run_dynamic_batch(self._error_f,*args,init_batch_size=self.error_batch_size)
            return np.sum([e[0] * e[1] for e in res],axis=0)
            # size = args[0].shape[0]
            # if size <= self.max_data_size:
            #     return self._error_f(*args)
            # e = 0
            # for batch in range(0,size,self.max_data_size):
            #     e += (self._error_f(*[a[batch:batch+self.max_data_size,:] if isinstance(a, jnp.ndarray) else a for a in args]) * min(size-batch,self.max_data_size))
            # return e/size
        
        def forward_f(self,*args):
            res,self.forward_batch_size = self._run_dynamic_batch(self._forward_f,*args,init_batch_size=self.forward_batch_size)
            return np.vstack([r[0] for r in res])
            # if args[0].shape[0] <= self.max_data_size:
            #     return self._forward_f(*args)
            # outs = []
            # for batch in range(0,args[0].shape[0],self.max_data_size):
            #     outs.append(self._forward_f(*[a[batch:batch+self.max_data_size,:] if isinstance(a, jnp.ndarray) else a  for a in args]))
            # return np.vstack(outs)

        def get_coef(self):
            return [np.array(p.get_coef()) for p in self.parameters.values()]
        
        def set_coef(self,*coef):
            [p.set_coef(c) for p,c in zip(self.parameters.values(),coef)]

        def add_dropout(self,p=0.1,name="dropout"):
            self.options[name] = Options.Dropout(p=p)
        
        def iter_batches(self,n_samples,error_f,test_f=None,n_batches=1,adaptive_rate=15,batch_size=-1,n_epochs=10000,
                         verbose=True,print_freq=1,use_best_Ws=True,early_exit=40,reset_checkpoint=False,shuffle_batches=True,**kwargs):
            if len(self.error_trace) == 0:
                self.error_trace.append(error_f())
                self.best_val = self.error_trace[-1][1]
                self.best_Ws = self.get_coef()
            
            if adaptive_rate > 0:
                early_exit = adaptive_rate
            best_epoch = 0
            checkpoint_val = self.best_val
            self.test_trace = []

            if n_batches > 0:
                batch_size = n_samples//n_batches
            if batch_size <= 0:
                batch_size = n_samples

            index = np.arange(n_samples)

            for i in range(n_epochs):

                for batch in range(0,n_samples,batch_size):
                    yield index[batch:batch+batch_size]
                    #yield (batch,batch+batch_size)
                if shuffle_batches:
                    np.random.shuffle(index)

                if not test_f is None:
                    tmp_ws = self.get_coef()
                    self.set_coef(*self.best_Ws)
                    self.test_trace.append(test_f())
                    self.set_coef(*tmp_ws)
                if error_f is None:
                    continue
                self.error_trace.append(error_f())
                if verbose and (i%print_freq==0 or i>=(n_epochs-print_freq)):
                    print(f'\t\tepoch {i}, train error={self.error_trace[-1][0]} val error={self.error_trace[-1][1]}')

                if use_best_Ws and self.error_trace[-1][1] < self.best_val:
                    if verbose:
                        print(f'\t\t\tnew best: {self.error_trace[-1][1]}')
                    self.best_val = self.error_trace[-1][1]
                    self.best_Ws = self.get_coef()
                    best_epoch = i
                if self.error_trace[-1][1] < checkpoint_val and reset_checkpoint:
                    checkpoint_val = self.error_trace[-1][1]
                    best_epoch = i
                    if verbose:
                        print(f'\t\t\treseting checkpoint: {self.error_trace[-1][1]}')


                if best_epoch < (i-early_exit) and early_exit > 0:
                    flag = True
                    if adaptive_rate > 0:
                        flag = False
                        for param in self.parameters.values():
                            if not param.optimizer.adapt_learning_rate(**kwargs):
                                flag = True
                                break
                        if not flag:
                            best_epoch = i
                            checkpoint_val = self.error_trace[-1][1]
                            if verbose:
                                print(f'\tadapting learning rate')
                    if flag:
                        if verbose:
                            print(f'\tearly exit after {i} epochs')
                        break
                        
            if use_best_Ws:
                self.set_coef(*self.best_Ws)

        def step(self,*args):
            for g,p in zip(self.grad_f(*args),self.parameters.keys()):
                self.parameters[p].step(g)
        
        def _convert_data(self,*data,**kwargs):
            return self.data_model.standardize(*data)
        
        def train(self,Xtr,Ttr,Xval,Tval,standardize=True,**kwargs):
            if standardize:
                Xtr,Ttr,Xval,Tval = self._convert_data(Xtr,Ttr,Xval,Tval)
            for index in self.iter_batches(Xtr.shape[0],lambda:[self.error_f(Xtr,Ttr),self.error_f(Xval,Tval)],**kwargs):
                self.step(Xtr[index],Ttr[index])
            return self
        
        def compile_model(self,loss_f,forward_f,**kwargs):
            # self.compiled = tools.compile_model(loss_f,forward_f,argnums=len(self.parameters),**kwargs)
            self.compiled = tools.compile_model(loss_f,forward_f,**{"argnums":[i for i,param in enumerate(self.parameters.values()) if param.requires_grad],**kwargs})

        ##loading/storing

        def load(self,path,model_dir=MODEL_DIRECTORY):
            self._load(load_file(path,directory=model_dir))
            return self

        def _load(self,config):
            self.params = config["settings"]
            self.initialize(**config["settings"])
            self.parameters = self._load_params(config["parameters"])
            return self


        def _load_params(self,sub_modules):
            loaders = {
                "module":lambda k,v:type(self.parameters[k])(self.data_model)._load(v),
                "container":lambda k,v:NNets.ParameterContainer.load(v),
                "parameters":lambda k,v:NNets.Parameters(**v)
            }
            return {k:loaders[v["type"]](k,v) for k,v in sub_modules.items() }
        
        def store(self,path,model_dir=MODEL_DIRECTORY):
            if not path.split(".")[-1] == "json":
                raise Exception("path must be .json")
            store_file(path,self._store(),directory=model_dir)
        
        def _store(self):
            return {"type":"module","settings":self.params,"parameters":{k:v._store() for k,v in self.parameters.items()}}
        

        ## federated learning

        def train_client(self,Xtr,Ttr,Xval,Tval,standardize=True,federated_param_names=[],independent_mode=True,**kwargs):
            if len(federated_param_names) <= 0:
                federated_param_names = list(self.parameters.keys())
            else:
                independent_mode = True
            if standardize:
                Xtr,Ttr,Xval,Tval = self._convert_data(Xtr,Ttr,Xval,Tval)
            for index in self.iter_batches(Xtr.shape[0],lambda:[self.error_f(Xtr,Ttr),self.error_f(Xval,Tval)],**(kwargs if independent_mode else {**kwargs,"use_best_Ws":False})):
                self.step(Xtr[index],Ttr[index])
                yield {k:v.get_coef() for k,v in self.parameters.items()}
            return self
        
        def train_server(self,aggregator,val_data=None,standardize=True,federated_param_names=[],independent_mode=True,**kwargs):
            if not independent_mode:
                if standardize:
                    Xval,Tval = self._convert_data(*val_data)
                validator = self.iter_batches(1,lambda:[-1,Xval,Tval],**kwargs)
            while aggregator:
                new_params = aggregator.step()
                for k,v in new_params.items():
                    self.parameters[k].set_coef(v)
                yield new_params
                if not independent_mode:
                    next(validator)
        
    class Regression(Module):
        def initialize(self,parameters = None,forward_f = lambda Ws,X:fc_layer(Ws,ACTIVATION["tanh"],X),loss_f="mse",options = {},**kwargs):
            self.parameters = NNets.Parameters.fully_connected(self.data_model.n_inputs,self.data_model.n_outputs,**kwargs) if parameters is None else parameters
            self.options = options
            self.compile_model(loss_f,forward_f,**kwargs)


        def use(self,X,standardize=True,unstandardize=True):
            Y = self.forward_f(self._convert_data(X) if standardize else X)
            #Y = self.forward_f(self.data_model.standardize(X) if standardize_input else X)
            return self.data_model.unstandardize(Y) if unstandardize else Y
        
        def __call__(self,X):
            return self.use(X,unstandardize=False,standardize=True)
        
        def _loss(self,Y,T):
            return np.mean(((T-Y)**2))
        
        def evaluate(self,Xte,Tte,standardize=True):
            Y = self.use(Xte,standardize=standardize)
            return self._loss(Y,Tte)
    

    class SelfSupervised(Regression):
        def evaluate(self,Xte,**kwargs):
            (Xte,Tte) = self._convert_data(Xte,**kwargs)
            return self.error_f(Xte,Tte)
            # Y = self.forward_f(Xte)
            # return self.error_f(Y,Tte)

        def train(self,Xtr,Xval,**kwargs):
            (Xtr,Ttr),(Xval,Tval) = self._convert_data(Xtr,Xval,**kwargs)
            return super().train(Xtr,Ttr,Xval,Tval,standardize=False,**kwargs)

    class Classifier(Regression):
        def initialize(self,forward_f = lambda Ws,X:softmax(fc_layer(Ws,ACTIVATION["relu"],X)),loss_f="nll",**kwargs):
            super().initialize(loss_f=loss_f,forward_f=forward_f,**kwargs)
        
        def evaluate(self,Xte,Tte,standardize=True):
            Yte = self.use(Xte,standardize=standardize)
            # print({d:np.mean(Yte[Tte==d] == d) for d in np.unique(Tte)})
            return np.mean([np.mean(Yte[Tte==d] == d) for d in np.unique(Tte)])
        
        # def evaluate(self,Xte,Tte):
        #     return np.mean(self.use(Xte) == Tte)

    class Interpreter(Classifier):
        def initialize(self,encoder_f=None,dec_act="relu",enc_outputs = 15,**kwargs):
            self.encoder_f = encoder_f
            super().initialize(
                parameters = {"classifier":NNets.Parameters.fully_connected(enc_outputs,len(self.data_model.labels),**kwargs)},
                loss_f = "nll",
                forward_f = lambda Ws,X: ACTIVATION["softmax"](fc_layer(Ws,ACTIVATION[dec_act],X).mean(axis=1))
            )
            self.max_data_size = MAX_DATA_SIZE * 1000

        def _convert_data(self,*data,**kwargs):
            data = self.data_model.standardize(*data) if len(data) > 1 else [self.data_model.standardize(*data)]
            cons = [
            jnp.vstack(
                [self.encoder_f(x[batch:batch+MAX_DATA_SIZE]) for batch in range(0,x.shape[0],MAX_DATA_SIZE)]
            ) 
                if x.shape[-1]==self.data_model.n_inputs 
                else x 
                for x in data
            ]
            return cons[0] if len(data)==1 else cons
            
class Functions:


    def convolution_1d(Ws,X,stride,padding=False):
        #X = (batch,rows,channel)
        #Ws = [(patch_size,channel, n_hiddens),(n_hiddens)]
        Ws,bias = Ws
        if padding:
            X = jnp.concatenate(
                (jnp.zeros((X.shape[0],Ws.shape[0],X.shape[-1])),X),
                axis = 1
            )
        Y = X[:,:-Ws.shape[0]:stride,:] @ Ws[0,:,:]
        for i in range(1,Ws.shape[0]):
            Y = Y + (X[:,i:(X.shape[1] - Ws.shape[0]) + i:stride,:] @ Ws[i,:,:])
        return Y + bias
    

    def convolution_2d(Ws,X,stride,padding=False):
        #X = (batch,rows,cols,channel)
        #Ws = [(patch_size,patch_size,channel, n_hiddens),(n_hiddens)]
        Ws,bias = Ws
        if padding: 
            X = jnp.concatenate(
                (jnp.zeros((X.shape[0],Ws.shape[0],X.shape[2],X.shape[3])),X),
                axis = 1
            )
            X = jnp.concatenate(
                (jnp.zeros((X.shape[0],X.shape[1],Ws.shape[1],X.shape[3])),X),
                axis = 2
            )
        Y = X[:,:-Ws.shape[0]:stride,:-Ws.shape[1]:stride,:] @ Ws[0,0,:,:]
        for i in range(1,Ws.shape[0]):
            for j in range(1,Ws.shape[1]):
                Y = Y + (X[:,i:(X.shape[1] - Ws.shape[0]) + i:stride,j:(X.shape[2] - Ws.shape[1]) + j:stride,:] @ Ws[i,j,:,:])
        return Y + bias
