import jax_nnets
import jax.numpy as jnp,jax,numpy as np
import math



def SSM(Ws,X):
    A,D,W_xdt = Ws
    W_x,W_dt = W_xdt
    dBC = X @ W_x
    B = dBC[:,:,:A.shape[1]]
    C = dBC[:,:,A.shape[1]:A.shape[1]*2]
    delta = dBC[:,:,A.shape[1]*2:]
    delta = jax_nnets.get_activation_f("softplus")(jax_nnets.linear_layer(W_dt,delta) if W_dt.ndim==2 else delta + W_dt.reshape((1,-1)))
    ## delta:(batch,seq,d_in); B,C:(batch,seq, d_state); A: (d_in,d_state)
    deltaA = jnp.exp(  jnp.expand_dims(delta,axis=-1) * -jnp.exp(A) )
    deltaBX = jnp.expand_dims(delta,axis=-1) * jnp.expand_dims(B,axis=-2) * jnp.expand_dims(X,axis=-1)
    H = jnp.zeros((X.shape[0],X.shape[2],A.shape[-1]))
    Y = jnp.zeros_like(X)
    ## H:(batch,d_in, d_state); deltaA:(batch,seq,d_in, d_state); deltaBX:(batch,seq,d_in, d_state); Y: (batch,seq,d_in)
    for i in range(X.shape[1]):
        H = deltaA[:,i] * H + deltaBX[:,i]
        Y = Y.at[:,i,:].set((H @ jnp.expand_dims(C[:,i],axis=-1)).squeeze(-1))
    return Y + (X * D)

def MAMBA_encode(emb_Ws,enc_Ws,X,act_func=jax.nn.silu,dropout=None):
    Y = jax_nnets.embedding_layer(emb_Ws,X)
    for i in range(len(enc_Ws)):
        input_Ws,conv_Ws,ssm_Ws,output_Ws,norm_Ws = enc_Ws[i]
        X = jax_nnets.linear_layer(input_Ws,jax_nnets.RMSnorm(Y,W=norm_Ws))
        X = jax_nnets.dropout_layer(X,dropout=dropout).reshape((Y.shape[0],Y.shape[1],-1,2))#.moveaxis([0,1,2,3],[3,0,1,2])#.swapaxes(0,3)
        Y1 = jax_nnets.Functions.convolution_1d(conv_Ws,X[:,:,:,0],padding=True,stride=1)
        Y1 = SSM(ssm_Ws,act_func(Y1))
        Y1 = Y1 * act_func(X[:,:,:,1])
        Y1 = jax_nnets.linear_layer(output_Ws,Y1)
        Y = Y1 + Y
    return jax_nnets.RMSnorm(Y)


class MAMBA(jax_nnets.NNets.Classifier):
    def get_block_parameters(self,model_dim,d_expand=2,d_state=16,patch_size=4,dt_rank_factor=16,**kwargs):
        d_in = model_dim * d_expand
        dt_rank =  math.ceil(d_in / max(dt_rank_factor,1))
        return jax_nnets.NNets.ParameterContainer(
            input_Ws = jax_nnets.NNets.Parameters((model_dim+1,d_in*2)),
            conv_Ws =  jax_nnets.NNets.Parameters([(patch_size,d_in, d_in),(d_in,)]),
            ssm_Ws = jax_nnets.NNets.ParameterContainer(
                A = jax_nnets.NNets.Parameters(shapes=(d_in,d_state),weights = np.repeat(np.log(np.arange(1,d_state+1)),d_in,axis=0)),
                D = jax_nnets.NNets.Parameters(shapes=(d_in,),weights = np.ones((d_in))),
                W_x = jax_nnets.NNets.Parameters([(d_in,d_state*2 + dt_rank),(dt_rank+1,d_in) if dt_rank_factor > 1 else (d_in)]),
            ),
            output_Ws = jax_nnets.NNets.Parameters((d_in+1,model_dim)),
            norm_Ws = jax_nnets.NNets.Parameters(shapes=(model_dim,),weights = np.ones((model_dim))),
        )


    def initialize(self, model_dim=40,n_blocks = 3,dropout=0.1,ssm_act="silu",dec_act="relu",pool_length=1,**kwargs):
        self.parameters = {
            "embedding":jax_nnets.NNets.Parameters.embedding(self.data_model,model_dim=model_dim),
            "encoder":jax_nnets.NNets.ParameterContainer(
                **{f'block{i}':self.get_block_parameters(model_dim,**kwargs) for i in range(n_blocks)}
            ),
            "decoder":jax_nnets.NNets.Parameters.fully_connected(model_dim,self.data_model.n_outputs)
        }
        self.add_dropout(p=dropout)
        self.compile_model(
            loss_f="nll",
            forward_f=lambda Ws,X,dropout=None:jax_nnets.softmax(
                jax_nnets.fc_layer(
                    Ws[2],
                    jax_nnets.get_activation_f(dec_act),
                    MAMBA_encode(Ws[0],Ws[1],X,act_func=jax_nnets.get_activation_f(ssm_act),dropout=dropout)[:,-pool_length:,:],
                    dropout=dropout
                ).mean(axis=1)
            )
        )
        


