import jax.numpy as jnp,jax,numpy as np
import jax_nnets


def LSTM_layer(Ws,X):
    H = jnp.zeros((X.shape[0],X.shape[2]))
    C = jnp.zeros((X.shape[0],X.shape[2]))
    Hs = jnp.zeros((X.shape[0],X.shape[1],H.shape[1]))
    for i in range(X.shape[1]):
        hc = jax_nnets.linear_layer(Ws.reshape((Ws.shape[0],Ws.shape[1])),jnp.concatenate((X[:,i,:],H),axis=1)).reshape((X.shape[0],4,H.shape[1])).swapaxes(0,1)
        fg = jax_nnets.ACTIVATION["sigmoid"](hc[0])
        ig = jax_nnets.ACTIVATION["sigmoid"](hc[1])
        og = jax_nnets.ACTIVATION["sigmoid"](hc[2])
        c_til = jax_nnets.ACTIVATION["tanh"](hc[3])
        C = (fg * C) + (ig * c_til)
        H = jax_nnets.normalize(og * jax_nnets.ACTIVATION["tanh"](C))
        Hs = Hs.at[:,i,:].set(H)
    return Hs

def LSTM_classifier(Ws,act_func,X,dropout=None):
    lstm_Ws,clf_Ws = Ws
    for i in range(len(lstm_Ws)):
        X = jax_nnets.dropout_layer(LSTM_layer(lstm_Ws[i],X),dropout=dropout)
    return jax_nnets.softmax(jax_nnets.fc_layer(clf_Ws,act_func,X[:,-1,:]))

class LSTM(jax_nnets.NNets.Classifier):
    def get_lstm_parameters(n_outputs,model_dim=40,n_lstm_layers=1,dec_hiddens=[40],**kwargs):
        return jax_nnets.NNets.Parameters((
            [((model_dim*2)+1,model_dim*4)]*n_lstm_layers,jax_nnets.tools.get_fc_shapes(model_dim,n_outputs,dec_hiddens)
        ))
    
    def set_parameters(self,model_dim=40,**kwargs):
        self.parameters = {
            "embedding":jax_nnets.NNets.Parameters(( 
                (self.data_model.n_inputs,model_dim), [(nl+1,model_dim) for nl in self.data_model.n_lookups] 
            )),
            "lstm":LSTM.get_lstm_parameters(self.data_model.n_outputs,model_dim=model_dim,**kwargs)
        }
    
    def initialize(self,act_func="relu",dec_dropout=0.1,**kwargs):
        self.set_parameters(**kwargs)
        self.options = {"dropout":jax_nnets.Options.Dropout(p=dec_dropout)}
        self.compile_model(
            jax_nnets.tools.LOSS_FUNCS["nll"],
            lambda Ws,X,dropout=None:LSTM_classifier(Ws[1],jax_nnets.get_activation_f(act_func),jax_nnets.embedding_layer(Ws[0],X),dropout=dropout)
        )