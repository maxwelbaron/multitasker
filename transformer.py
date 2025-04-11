import jax.numpy as jnp,jax,numpy as np
import jax_nnets




def transformer_encoder(enc_Ws, act_func, X, PE, scores = None, k=1, dropout = None):
    emb_Ws,enc_Ws = enc_Ws
    X = jax_nnets.embedding_layer(emb_Ws,X,pos_enc=PE)
    enc_Ws = jax_nnets.Pruning.apply_pruning(enc_Ws,scores=scores,k=k)
    scale = jnp.sqrt(enc_Ws[0][0][0].shape[-1])
    
    for i in range(len(enc_Ws)):
        W_attention, W_fc = enc_Ws[i]

        W_keys,W_queries,W_values,W_combine = W_attention
        K = jax_nnets.linear_layer(W_keys.reshape((W_keys.shape[0],-1)),X).reshape((X.shape[0],X.shape[1],-1,W_keys.shape[-1])).swapaxes(1,2)
        Q = jax_nnets.linear_layer(W_queries.reshape((W_queries.shape[0],-1)),X).reshape((X.shape[0],X.shape[1],-1,W_queries.shape[-1])).swapaxes(1,2)
        V = jax_nnets.linear_layer(W_values.reshape((W_queries.shape[0],-1)),X).reshape((X.shape[0],X.shape[1],-1,W_values.shape[-1])).swapaxes(1,2)
        QKV = jax_nnets.dropout_layer(jax_nnets.softmax( (Q @ K.swapaxes(2,3))/scale ),dropout=dropout) @ V
        attention = jax_nnets.linear_layer(W_combine, QKV.swapaxes(1,2).reshape((X.shape[0],X.shape[1],-1)))
        
        X = jax_nnets.normalize(jax_nnets.dropout_layer(attention,dropout=dropout) + X)
        X = jax_nnets.normalize(jax_nnets.dropout_layer(jax_nnets.fc_layer(W_fc,act_func,X,dropout=dropout),dropout=dropout) + X)

    return X

class Transformer:
    pass


class ClozeTransformer(jax_nnets.NNets.SelfSupervised):
    def encoder_parameters(model_dim=40,depth=16,attention_blocks=3,heads_per_block=8,n_hiddens=[80],**kwargs):
        return jax_nnets.NNets.Parameters(
            [( [(model_dim+1,heads_per_block,depth)] * 3 +[((depth*heads_per_block)+1,model_dim)],jax_nnets.tools.get_fc_shapes(model_dim,model_dim,n_hiddens) )] * attention_blocks
        )
    
    def embedding_parameters(self,model_dim=40):
        return jax_nnets.NNets.Parameters(( 
            (self.data_model.n_inputs,model_dim), [(nl+1,model_dim) for nl in self.data_model.n_lookups] 
        ))
    
    def initialize(self,model_dim=40, act_func = "relu", mask_rate=0.2,dropout=0.1, **kwargs):
        self.parameters = {
            "encoder":jax_nnets.NNets.ParameterContainer(
                embedding = self.embedding_parameters(model_dim=model_dim),
                encoder = ClozeTransformer.encoder_parameters(model_dim=model_dim,**kwargs)
            ),
            "decoder":jax_nnets.NNets.Parameters.fully_connected(model_dim,self.data_model.n_embeddings, **kwargs)
        }
        self.options = {
            "dropout":jax_nnets.Options.Dropout(p=dropout),
            "mask_gen":jax_nnets.Options.RandomGen(p=mask_rate)
        }
        self.PE = jax_nnets.tools.get_positional_encoding(model_dim,seq_length=self.data_model.max_seq_length)

        def forward_f(Ws,X,mask_gen=None,dropout=None):
            mask = jax.random.binomial(mask_gen[0],1,1-mask_gen[1],shape=(X.shape[0],X.shape[1],1))
            Y = jax_nnets.fc_layer(Ws[1],jax_nnets.get_activation_f(act_func),transformer_encoder(Ws[0],jax_nnets.get_activation_f(act_func),X*mask,self.PE,dropout=dropout)).reshape((X.shape[0]*X.shape[1],-1))
            start = 0
            Y_lookup = []
            for nl in self.data_model.n_lookups:
                Y_lookup.append(jax_nnets.softmax(Y[:,start:start+nl]))
                #Y = Y.at[:,:,start:start+nl].set(jax_nnets.softmax(Y[:,:,start:start+nl]))
                start += nl
            return Y[:,:self.data_model.n_inputs], Y_lookup, (1 - mask.flatten()).reshape((1,Y.shape[0]))
        
        def loss_f(Y,T):
            Y_st,Y_rel,mask = Y
            T = T.reshape((Y_st.shape[0],-1))#[mask == 0,:] 
            n_elements = np.sum(mask) * (self.data_model.n_inputs+len(Y_rel))
            jnp.mean((Y_st - T[:,:self.data_model.n_inputs])**2)
            # print(np.mean(mask),(mask @ (Y_st - T[:,:model.data_model.n_inputs])**2).shape)
            st_loss = jnp.sum(mask @ (Y_st - T[:,:self.data_model.n_inputs])**2) * (self.data_model.n_inputs/n_elements)
            for i in range(len(Y_rel)):
                st_loss -= jnp.sum(mask @ jnp.log(jnp.take(Y_rel[i], T[:,self.data_model.n_inputs+i].astype(jnp.int16)-1 ,axis=0,mode="clip"))) * (1/(n_elements*Y_rel[i].shape[-1]))
                # print((mask @ jnp.log(jnp.take(Y_rel[i], T[:,model.data_model.n_inputs+i].astype(jnp.int16)-1 ,axis=0,mode="clip"))).shape)
            return st_loss

        self.compile_model(
            loss_f = loss_f,
            forward_f = forward_f
        )


        