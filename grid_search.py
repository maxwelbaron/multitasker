import file_manager,os,pandas,itertools,numpy as np,time
from data_manager import DataManager
from LSTM import LSTM

MODELS = {
    "LSTM":LSTM
}


def grid_search(model_type="LSTM",n_folds=3,dataset="IoT",features="all",result_dir=file_manager.RESULT_DIRECTORY,**kwargs):
    path = f'{dataset}_{features}_gridsearch_{model_type}.csv'
    metric_names = ["recall","accuracy","train_recall","time(s)","size(KB)","epochs","data(KB)"]
    loaded = os.path.exists(f'{result_dir}/{path}')
    results = file_manager.load_file(path,directory=result_dir) if loaded else pandas.DataFrame(columns=[*metric_names,*list(kwargs.keys())])
    params = [{k:v for k,v in zip(kwargs.keys(),combo)} for combo in itertools.product(*kwargs.values())]

    if loaded:
        for k in params[0].keys():
            if not k in results.columns:
                results.insert(len(results.columns),k,"default")

    for i in range(len(params)):
        print(f'[{i+1}/{len(params)}]: {params[i]}')
        if loaded and np.any(np.all([results[k].values == type(results[k][0])(v) for k,v in params[i].items()],axis=0)):
            print("skipping")
            continue

        data_manager = DataManager(dataset=dataset,features=features,**params[i])
        metrics = {k:0 for k in metric_names}
        
        for fold in range(n_folds):
            print(f'\tfold [{fold+1}/{n_folds}]')
            (Xtr,Ttr),(Xval,Tval),(Xte,Tte),data_model = data_manager.get_data(iteration=fold)
            model = MODELS[model_type](data_model,**params[i])

            start_time = time.perf_counter()
            model.train(Xtr,Ttr,Xval,Tval,**params[i])
            metrics["time(s)"] += (time.perf_counter() - start_time)
            Yte = model.use(Xte,standardize=True)
            metrics["recall"] += np.mean([np.mean(Yte[Tte==d] == d) for d in np.unique(Tte)])
            metrics["accuracy"] += np.mean(Yte==Tte)
            metrics["epochs"] += len(model.error_trace)
            metrics["size(KB)"] += (model.size()/1_000)
            metrics["data(KB)"] += (Xtr.nbytes / 1_000)
            Ytr = model.use(Xtr,standardize=True)
            metrics["train_recall"] += np.mean([np.mean(Ytr[Ttr==d] == d) for d in np.unique(Tte)])
    
        results.loc[len(results)] = {**{m:v/n_folds for m,v in metrics.items()},**params[i]}
        file_manager.store_file(path,results,directory=result_dir)
        print(results.loc[len(results)-1])
        print("\n\n")
    return results


grid_search(
    model_type="LSTM",
    features = "all",
    dataset = "IoT",
    adaptive_rate = [15],
    reset_checkpoint = [False],
    learning_rate = [0.01],
    n_batches = [1],
    lr_decay_factor = [3],
    lr_early_exit = [10e-6],
    n_epochs = [3_000],

    n_lstm_layers = [1,2],
    dec_hiddens = [[80],[40],[]],
    model_dim = [40,80],
    dec_dropout = [0.0,0.1]
)