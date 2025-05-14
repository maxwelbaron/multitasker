import numpy as np, pandas, os,jax,jax.numpy as jnp
from jax_nnets import FLOAT_T,tools
from file_manager import load_file

DATA_DIRECTORY = "./data"

def load_dataset(dataset="IoT",class_names=[],max_packets=25_000,min_packets=1_000,**kwargs):
    catalog = load_file(f"{dataset}/catalog.json",directory=DATA_DIRECTORY)
    df_list = []
    if len(class_names) < 1:
        class_names = list(catalog.keys())
    for k,v in catalog.items():
        if (v["total_packets"] < min_packets) or (not k in class_names):
            # print(f'removing {k}')
            continue
        df = load_file(v["path"],directory=DATA_DIRECTORY)
        df.insert(0,"class_name",k)
        df_list.append(df.iloc[:max_packets,:])
    return pandas.concat(df_list,ignore_index=True)

MAX_SEQ_LENGTH = 75

def get_partitions(labels,random_state=42,te_size=2,val_size=1,tr_size=7,**kwargs):
    card = te_size + val_size + tr_size
    index = np.arange(labels.shape[0])
    np.random.seed(random_state)
    np.random.shuffle(index)
    indices = [index[labels[index] == l] for l in np.unique(labels)]
    def subset(start):
        s = np.hstack([l[(l.shape[0]//card)*start:(l.shape[0]//card)*(1+start)] for l in indices])
        np.random.shuffle(s)
        return s
    partition = [subset(i) for i in range(card)]
    _iteration = [0]
    def get_partition(iteration=None):
        if iteration is None:
            iteration = _iteration[0]
            _iteration[0] += 1
        iteration *= te_size
        shifted = partition[iteration%card:] + partition[:iteration%card]
        return np.hstack(shifted[:tr_size]),np.hstack(shifted[tr_size:tr_size+val_size]),np.hstack(shifted[tr_size+val_size:])
    return get_partition

class DataManager:
    DATASETS = {
        "all":{},
        "standard":{"relative_features":[]},
        "relative":{"standard_features":[]},
        "debug":{"class_names":["Amazon","Arlo","Eye"],"max_packets":10000}
    }

    class DataModel:
        def __init__(self,n_inputs,n_lookups,class_names,max_seq_length,x_dtype=FLOAT_T):
            self.n_inputs,self.n_lookups,self.class_names,self.max_seq_length,self.x_dtype = n_inputs,n_lookups,np.array(class_names),max_seq_length,x_dtype
            self.n_input_features = self.n_inputs + len(self.n_lookups)
            self.n_embeddings = self.n_inputs + np.sum(self.n_lookups)
            self.lookups = [np.arange(l).reshape((1,-1)) for l in self.n_lookups]
            self.n_outputs = class_names.shape[0]

        def make_indicators(self,X):
            # return [np.where(X[:,:,i+self.n_inputs:i+1+self.n_inputs] == self.lookups[i],1,-1) for i in range(len(self.n_lookups))]
            return jnp.concatenate([ X[:,:,:self.n_inputs],*[np.where(X[:,:,i+self.n_inputs:i+1+self.n_inputs] == self.lookups[i],1,-1) for i in range(len(self.n_lookups))] ],axis=-1)

        def standardize(self,*data):
            converted = [jnp.array(d,dtype=self.x_dtype) if d.ndim>=3 else jnp.array((d==self.class_names).astype(int),dtype=jnp.int8) for d in data]
            return converted[0] if len(data)==1 else converted
        def unstandardize(self,*data):
            converted = [x if x.ndim>=3 else self.class_names[(np.argmax(x,axis=1),None)] for x in data]
            return converted[0] if len(data)==1 else converted
    
    def _load_data(self,dataset="IoT",relative_features=["IPsrc","IPdst","srcPrt","dstPrt"],
                 standard_features=["subnetSrc","subnetDst","broadcastDst","TLS","TCP","UDP","HTTP","HTTPS","hdrLen","appLen","appEntropy","commonSrc","commonDst","time_dif"],**kwargs):
        data = load_dataset(dataset=dataset,**kwargs)
        self.T = data.iloc[:,0].values
        self.raw_data = data[relative_features].values if len(relative_features) > 0 else np.zeros((0,0))
        if len(standard_features) > 0:
            self.X = data[standard_features].values
            self.unstandardized_features = np.where((np.amin(self.X,axis=0) < -1) | (np.amax(self.X,axis=0) > 1))[0]
        else:
            self.X = np.zeros((0,0))
            self.unstandardized_features = []
        self.labels = np.unique(self.T)
        print(self.labels)

    def __init__(self,features="all",**kwargs):
        self._load_data(**DataManager.DATASETS[features],**kwargs)
        self.init_partitions(**kwargs)

    def __repr__(self):
        return f'DataManager: \n\tshape: {self.X.shape}\n\tdevices: {self.labels}'

    def __getitem__(self,index):
        X = self.X_w[index,:]
        X[:,:,self.unstandardized_features] = (X[:,:,self.unstandardized_features] - self.c_means) / self.c_stds
        return (X,self.T_w[index])
    
    def _split_data(self,iteration=None):
        self.tr_i,self.val_i,self.te_i = self.partition_generator(iteration=iteration)
        c_features = self.X_w[self.tr_i][:,:,self.unstandardized_features].reshape((self.tr_i.shape[0]*self.X_w.shape[1],-1))
        self.c_means = np.mean(c_features,axis=0)
        self.c_stds = np.std(c_features,axis=0)


    def init_partitions(self,window_size=32,unknown_lookups=True,**kwargs):
        window_size = min(window_size,MAX_SEQ_LENGTH)
        windows = []
        window_labels = []
        index_list = []
        for d in self.labels:
            indices = np.where(self.T == d)[0]
            for i in range(0,1+(len(indices)-window_size)):
                index_list.append(indices[i:i+window_size])
                if self.X is not None:
                    windows.append(self.X[indices[i:i+window_size],:])
                window_labels.append(d)
            # index_list.append(indices)
            # device_windows = np.zeros((1+(len(indices)-window_size),window_size,self.X.shape[1]))
            # for i in range(window_size):
            #     device_windows[:,i,:] = self.X[indices[i:indices.shape[0]-(window_size-i-1)]]
            # window_labels += [d]*len(device_windows)
            # windows.append(device_windows)
        self.T_w = np.expand_dims(np.array(window_labels),-1)
        X_w = np.stack(windows,axis=0)
        if self.raw_data is None:
            self.X_w = X_w
            self.data_model = DataManager.DataModel(X_w.shape[-1], [], self.labels,window_size)
        else:
            raw_features = np.zeros((X_w.shape[0],window_size,self.raw_data.shape[1]))
            for window_i,data_i in enumerate(index_list):
                for j in range(self.raw_data.shape[1]):
                    local_vals = np.unique(self.raw_data[data_i,j])
                    raw_features[window_i,:,j] = np.where(self.raw_data[data_i,j:j+1] == local_vals)[1]
            if unknown_lookups:
                raw_features += 1
            self.X_w = np.concatenate((X_w,raw_features),axis=-1)
            self.data_model = DataManager.DataModel(X_w.shape[-1], [int(np.amax(raw_features[:,:,j])) for j in range(self.raw_data.shape[1])], self.labels,window_size)
        self.partition_generator = get_partitions(self.T_w[:,0],**kwargs)
        self._split_data()
    
    def get_data(self,change_split=True,iteration=None):
        if change_split or iteration is not None:
            self._split_data(iteration=iteration)
        Xtr,Ttr = self[self.tr_i]
        return (Xtr,Ttr),self[self.val_i],self[self.te_i],self.data_model

# class DataCollection:
#     def __init__(self,*dms):
#         self.dm_list = dms

#     def _split_data(self,iteration=None):
#         [dm._split_data(iteration=iteration) for dm in self.dm_list]

    
