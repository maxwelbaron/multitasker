import numpy as np,os,pandas,time,json,copy

## globals 

RESULT_DIRECTORY = "./results"
MAX_SIZE = 1500000
MODEL_DIRECTORY = "./saved_models"


## helper functions

get_path = lambda path,directory:(path,path.split(".")[-1]) if directory is None else (f'{directory}/{path}',path.split(".")[-1])

def load_file(fname,directory=None):
    path,ext = get_path(fname,directory)
    if ext == "csv":
        return pandas.read_csv(f'{path}').fillna("None")
    with open(path,'r') as f:
        return json.load(f)

def delete_file(fname,directory=None):
    path,_ = get_path(fname,directory)
    os.remove(path)
    
def store_file(fname,data,directory=None):
    path,ext = get_path(fname,directory)
    directory = "/".join(path.split("/")[:-1])
    if not os.path.exists(directory):
        print(f'creating directory: {directory}')
        os.mkdir(directory)
    if ext == "csv":
        if type(data) != pandas.DataFrame:
            data = pandas.DataFrame(data)
        return data.to_csv(path,index=False,na_rep="None")
    with open(path,'w') as f:
        return json.dump({k:v.tolist() if hasattr(v,"tolist") else v for k,v in data.items()},f,indent=4,default = lambda obj:obj.item())
    
def access_locked_file(fname,write_lock=True,wait_time=5.0,directory = RESULT_DIRECTORY):
    while True:
        file = load_file(fname,directory=directory)
        if file["write_lock"]:
            print(f"{fname} locked, waiting")
            time.sleep(wait_time)
            continue
        file["write_lock"] = write_lock
        store_file(fname,file,directory=directory)
        break
    file["write_lock"] = False
    return file

## setup

if not os.path.exists(MODEL_DIRECTORY):
    os.mkdir(MODEL_DIRECTORY)
    store_file("model_configs.json",{"write_lock":False},directory=MODEL_DIRECTORY)

if not os.path.exists(RESULT_DIRECTORY):
    os.mkdir(RESULT_DIRECTORY)
        

## result files

class LockManager:
    def __init__(self,lock_fname = "experiments"):
        self.lock_fname = lock_fname + "_locks.json"
        self.lock_dir = f'{RESULT_DIRECTORY}/{lock_fname}_files'
        if not os.path.exists(self.lock_dir):
            os.mkdir(self.lock_dir)
            store_file(self.lock_fname,{"write_lock":False,"locks":{},"params":{}},directory=RESULT_DIRECTORY)
        self.open_files = {}

    def _open_lock_manager(self,write_lock=True):
        self.lock_file = access_locked_file(self.lock_fname,write_lock=write_lock)
    
    def _close_lock_manager(self):
        self.lock_file["write_lock"] = False
        store_file(self.lock_fname,self.lock_file,directory=RESULT_DIRECTORY)

    def abort(self,error=Exception("error")):
        self._close_lock_manager()
        raise error
    
    def release_locks(self,*files):
        self._open_lock_manager()
        for f in files:
            self.lock_file["locks"][f] = False
        self._close_lock_manager()

    def delete_file(self,fname):
        self._open_lock_manager()
        try:
            delete_file(fname,directory=self.lock_dir)
            self.lock_file["locks"].pop(fname)
            self.lock_file["params"].pop(fname)
        except Exception as e:
            self.abort(error=e)
        self._close_lock_manager()

    def exists(self,fname):
        return os.path.exists(f'{self.lock_dir}/{fname}')
    
    def __enter__(self):
        self.open_files = {}

    def __exit__(self,exc_type,exc_value,traceback):
        if exc_type != None:
            for k,v in self.open_files.items():
                self.update_file(k,v)

    def create_file(self,fname,data,**params):
        self._open_lock_manager()
        try:
            self.lock_file["locks"][fname] = False
            self.lock_file["params"][fname] = params
            store_file(fname,data,directory=self.lock_dir)
        except Exception as e:
            self.abort(error=e)
        self._close_lock_manager()

    def open_file(self,fname,write_lock=True,wait_time=5.0):
        self._open_lock_manager()
        try:
            while self.lock_file["locks"][fname]:
                self._close_lock_manager()
                time.sleep(wait_time)
                print(f"{fname} locked, waiting")
                self._open_lock_manager()
                wait_time *= 1.5
                if wait_time > 10000:
                    raise Exception("timeout")

            self.lock_file["locks"][fname] = write_lock
            file = load_file(fname,directory=self.lock_dir)
        except Exception as e:
            self.abort(error=e)
        self.open_files[fname] = copy.deepcopy(file)
        self._close_lock_manager()
        return file
    
    
    def read_file(self,fname):
        return self.open_file(fname,write_lock=False)
    
    def update_file(self,fname,data,release_lock=True):
        self._open_lock_manager()
        try:
            self.lock_file["locks"][fname] = not release_lock
            store_file(fname,data,directory=self.lock_dir)
        except Exception as e:
            self.abort(error=e)
        self.open_files[fname] = copy.deepcopy(data)
        self._close_lock_manager()
