import numpy as np
import os
import csv
import random
from numpy.lib.stride_tricks import as_strided

def gen_QAM_constellation(n):
    #what about the distance d between points?
    #n=16 -> -3, -1, +1, +3
    constellation = np.linspace(int(-np.sqrt(n)+1), int(np.sqrt(n)-1), int(np.sqrt(n)))
    #n=16 -> sqrt((9+1+1+9)/4)
    alpha = np.sqrt((constellation ** 2).mean())
    constellation /= (alpha*np.sqrt(2))
    return constellation


def convert_from_decibel(db):
    return 10 ** (db/10)

def csv_to_dict(path, data_dict=None):

    if data_dict == None:
        data_dict = {}
    with open(path, mode='r') as infile:
        reader = csv.reader(infile)
        first_row = True
        for rows in reader:
            if first_row:
                col_names = rows
                first_row = False
            col_ct = 0
            for name in col_names:
                try:
                    tmp_float = float(rows[col_ct])
                    is_integer = tmp_float.is_integer()
                    if is_integer:
                        data_dict[name] = int(tmp_float)
                    else:
                        data_dict[name] = tmp_float
                except:

                    data_dict[name] = rows[col_ct]

                col_ct += 1

    return data_dict





def write_params_csv(path, params):

    with open(os.path.join(path, 'model_params.csv'), 'w') as csvfile:

        header = [str(key) for key in params.keys()]

        wr = csv.DictWriter(csvfile, fieldnames=header)

        wr.writeheader()

        wr.writerow(params)

    return

class DataLoader(object):
    def __init__(self,config):
        self.config = config
        self.prepare_data()
        self.cR = config["cR"]
        self.cT = config["cT"]
        self.batch_size = 64
        self.reduced_set = False

    def toeplitz(self,c, r=None):
        """
        Construct a Toeplitz matrix.
        The Toeplitz matrix has constant diagonals, with c as its first column
        and r as its first row. If r is not given, ``r == conjugate(c)`` is
        assumed.
        Parameters
        ----------
        c : array_like
            First column of the matrix.  Whatever the actual shape of `c`, it
            will be converted to a 1-D array.
        r : array_like, optional
            First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
            in this case, if c[0] is real, the result is a Hermitian matrix.
            r[0] is ignored; the first row of the returned matrix is
            ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
            converted to a 1-D array.
        Returns
        -------
        A : (len(c), len(r)) ndarray
            The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.
        See Also
        --------
        circulant : circulant matrix
        hankel : Hankel matrix
        solve_toeplitz : Solve a Toeplitz system.
        Notes
        -----
        The behavior when `c` or `r` is a scalar, or when `c` is complex and
        `r` is None, was changed in version 0.8.0. The behavior in previous
        versions was undocumented and is no longer supported.
        Examples
        --------
        >>> from scipy.linalg import toeplitz
        >>> toeplitz([1,2,3], [1,4,5,6])
        array([[1, 4, 5, 6],
               [2, 1, 4, 5],
               [3, 2, 1, 4]])
        >>> toeplitz([1.0, 2+3j, 4-1j])
        array([[ 1.+0.j,  2.-3.j,  4.+1.j],
               [ 2.+3.j,  1.+0.j,  2.-3.j],
               [ 4.-1.j,  2.+3.j,  1.+0.j]])
        """
        c = np.asarray(c).ravel()
        if r is None:
            r = c.conjugate()
        else:
            r = np.asarray(r).ravel()
        # Form a 1-D array containing a reversed c followed by r[1:] that could be
        # strided to give us toeplitz matrix.
        vals = np.concatenate((c[::-1], r[1:]))
        out_shp = len(c), len(r)
        n = vals.strides[0]
        return as_strided(vals[len(c) - 1:], shape=out_shp, strides=(-n, n)).copy()

    def correlate_iid_channel(self,H,normalization=True,cR=0.1,cT=0.1):
        """ Takes a Gaussian i.i.d channel matrix is input and computes the
        correlated channel according to the exponential correlation matrix model"""
    
        nr_rx = self.n_receiver
        nr_tx = self.n_sender
        # compute basis vectors
        r_vec = cR**np.arange(1,(nr_rx +1 ) )
        t_vec = cT**np.arange(1,(nr_tx +1 ) )
    
        R_r = self.toeplitz(r_vec) ** (1/2)
        R_t = self.toeplitz(t_vec) ** (1/2)
    
        zeros_r = np.zeros((nr_rx, nr_rx))
        zeros_t = np.zeros((nr_tx, nr_tx))
    
        # Create the block diagonal matrix for using with the stacked channel matrix
        R_r_upper = np.concatenate((R_r, zeros_r), axis=1)
        R_r_lower = np.concatenate((zeros_r, R_r), axis=1)
        R_r = np.concatenate((R_r_upper, R_r_lower), axis=0)
    
        R_t_upper = np.concatenate((R_t, zeros_t), axis=1)
        R_t_lower = np.concatenate((zeros_t, R_t), axis=1)
        R_t = np.concatenate((R_t_upper, R_t_lower), axis=0)
    
        H_c = np.dot( np.dot(R_r, H), R_t )
        if normalization is True:
            # normalize every column of H which is the same as normalizing the
            # complex values itself if we sum up Re(1)^2+Im(1)^2+... anyway
            # h_i = { h_{ij} / \sqrt{ \sum h_{ij}^2  } }
            H_c = H_c / np.sqrt(np.sum(H_c**2,axis=0))
        return H_c

    def filter_files(self,files,min_idx,max_idx):
        filtered_files = []
        for f in files:
            drop = int(f.split("_")[-2])
            if drop >= min_idx and drop<max_idx:
                filtered_files.append(f)
        return filtered_files
        
    def prepare_data(self):

        train_data = {"H":[],
                    "y":[],
                    "x":[],
                    "QAM_constellation":[],
                    "QAM_order":[],
                    "nr_rx":[],
                    "nr_tx":[],
                    "snr":[]}
        val_data = {"H":[],
                    "y":[],
                    "x":[],
                    "QAM_constellation":[],
                    "QAM_order":[],
                    "nr_rx":[],
                    "nr_tx":[],
                    "snr":[]}

        test_data = {"H":[],
                    "y":[],
                    "x":[],
                    "QAM_constellation":[],
                    "QAM_order":[],
                    "nr_rx":[],
                    "nr_tx":[],
                    "snr":[]}
        all_files = os.listdir(self.config["data_path"])
        db = self.config["data_path"].split("/")[-1]
        db = db.split("dB")[0]
        self.db = float(db)
        has_additional_test_data = "additional_test_data" in all_files
        all_files = [f for f in all_files if not("additional_test_data" in f) and not("results" in f)]
        if has_additional_test_data:
            train_percentage = self.config['train_percentage'] * 2
            val_percentage = self.config['val_percentage']
        else:
            train_percentage = self.config['train_percentage']
            val_percentage = self.config['val_percentage']

        if "3GPP" in self.config["data_path"]:
            random.seed(10)
            ndrops = int(all_files[-1].split("_")[-2])+1
            train_files = self.filter_files(all_files,0,ndrops//2)
            random.shuffle(train_files)
            val_files = self.filter_files(all_files,ndrops//2,ndrops//2+ndrops//4)
            random.shuffle(val_files)
            test_files = self.filter_files(all_files,ndrops//2+ndrops//4,ndrops+1)
            random.shuffle(test_files)

            all_files = train_files+val_files+test_files

        N = len(all_files)
        h_path = self.config["data_path"].split("/")[:-1]
        h_path = "/".join(h_path)
        h_path = h_path.split("_")[:-1]
        h_path = h_path + ["H"]
        h_path = "_".join(h_path)
        for i_f,f in enumerate(all_files):
            data = np.load(os.path.join(self.config["data_path"],f),allow_pickle=True)
            if "3GPP" in self.config["data_path"]:
                data_h = np.load(os.path.join(h_path,f),allow_pickle=True)
            else:
                data_h = data

            if 'arr_0' in data.keys():
                data = data['arr_0'].item()
            if 'arr_0' in data_h.keys():
                data_h = data_h['arr_0'].item()

            if i_f<train_percentage*N:
                H = data_h['H']
                if len(H.shape)==2:
                    H = np.expand_dims(H,0)
                train_data["H"].append(H)
                y = data['y']
                if len(y.shape)==2:
                    y = np.expand_dims(y,0)
                train_data["y"].append(y)
                x = data['x']
                if len(x.shape)==2:
                    x = np.expand_dims(x,0)
                train_data["x"].append(x)
                train_data["QAM_constellation"].append(data['QAM_constellation'])
                train_data["QAM_order"].append(data['QAM_order'])
                train_data["nr_rx"].append(data['nr_rx'])
                train_data["nr_tx"].append(data['nr_tx'])
                train_data["snr"].append(data['snr'])
            elif i_f >=train_percentage*N and i_f <(train_percentage+val_percentage)*N:
                H = data_h['H']
                if len(H.shape)==2:
                    H = np.expand_dims(H,0)
                val_data["H"].append(H)
                y = data['y']
                if len(y.shape)==2:
                    y = np.expand_dims(y,0)
                val_data["y"].append(y)
                x = data['x']
                if len(x.shape)==2:
                    x = np.expand_dims(x,0)
                val_data["x"].append(x)
                val_data["QAM_constellation"].append(data['QAM_constellation'])
                val_data["QAM_order"].append(data['QAM_order'])
                val_data["nr_rx"].append(data['nr_rx'])
                val_data["nr_tx"].append(data['nr_tx'])
                val_data["snr"].append(data['snr'])
            else:
                H = data_h['H']
                if len(H.shape)==2:
                    H = np.expand_dims(H,0)
                test_data["H"].append(H)
                y = data['y']
                if len(y.shape)==2:
                    y = np.expand_dims(y,0)
                test_data["y"].append(y)
                x = data['x']
                if len(x.shape)==2:
                    x = np.expand_dims(x,0)
                test_data["x"].append(x)
                test_data["QAM_constellation"].append(data['QAM_constellation'])
                test_data["QAM_order"].append(data['QAM_order'])
                test_data["nr_rx"].append(data['nr_rx'])
                test_data["nr_tx"].append(data['nr_tx'])
                test_data["snr"].append(data['snr'])

        if has_additional_test_data:
            all_files = os.listdir(os.path.join(self.config["data_path"],"additional_test_data"))
        
            for i_f,f in enumerate(all_files):
                data = np.load(os.path.join(self.config["data_path"],"additional_test_data",f))
                test_data["H"].append(data['H'])
                test_data["y"].append(data['y'])
                test_data["x"].append(data['x'])
                test_data["QAM_constellation"].append(data['QAM_constellation'])
                test_data["QAM_order"].append(data['QAM_order'])
                test_data["nr_rx"].append(data['nr_rx'])
                test_data["nr_tx"].append(data['nr_tx'])
                test_data["snr"].append(data['snr'])

        self.n_receiver = train_data["y"][0].shape[-2]//2
        self.n_sender = train_data["x"][0].shape[-2]//2

        ml_ser = 0
        mmse_ser = 0
        bp_ser = 0
        loaded_ml = False
        loaded_bp = False
        try:
            #Try load baseline results
            for i_f,f in enumerate(all_files):
                data_mmse = np.load(os.path.join(self.config["data_path"],f.replace("channel","results_mmse")))
                mmse_ser += np.mean(data_mmse["SE"])
                if os.path.exists(os.path.join(self.config["data_path"],f.replace("channel","results_ml"))):
                    data_ml = np.load(os.path.join(self.config["data_path"],f.replace("channel","results_ml")))
                    ml_ser += np.mean(data_ml["SE"])
                    loaded_ml = True
                if os.path.exists(os.path.join(self.config["data_path"],f.replace("channel","results_bp"))):    
                    data_bp = np.load(os.path.join(self.config["data_path"],f.replace("channel","results_bp")))
                    bp_ser += np.mean(data_bp["SE"])
                    loaded_bp = True
                    
            mmse_ser = mmse_ser/len(all_files)
            print("MMSE SER = "+str(mmse_ser))
            
            if loaded_bp:
                bp_ser = bp_ser/len(all_files)
                print("BP SER = "+str(bp_ser))
            if loaded_ml:
                ml_ser = ml_ser/len(all_files)
                print("ML SER = " +str(ml_ser))
            
        except:
            print("No baseline results found")

        print("Concatenating data...")
        train_data["H"] = np.concatenate(train_data["H"],axis=0)
        H_complex = train_data["H"][:,:self.n_receiver,:self.n_sender]+1j*train_data["H"][:,self.n_receiver:,:self.n_sender]
        H_complex_conj = train_data["H"][:,:self.n_receiver,:self.n_sender]-1j*train_data["H"][:,self.n_receiver:,:self.n_sender]
        train_data["y"] = np.concatenate(train_data["y"],axis=0)
        train_data["x"] =  np.concatenate(train_data["x"],axis=0)

        x_tiled = np.tile(np.expand_dims(np.squeeze(train_data["x"]), axis=1), [1,train_data["H"].shape[1],1])
        y_clean = np.sum(x_tiled * train_data["H"], axis=-1,keepdims=True)
        train_data["y_clean"] = y_clean

        val_data["H"] = np.concatenate(val_data["H"],axis=0)
        val_data["y"] = np.concatenate(val_data["y"],axis=0)
        val_data["x"] =  np.concatenate(val_data["x"],axis=0)
        x_tiled = np.tile(np.expand_dims(np.squeeze(val_data["x"]), axis=1), [1,val_data["H"].shape[1],1])
        y_clean = np.sum(x_tiled * val_data["H"], axis=-1,keepdims=True)
        val_data["y_clean"] = y_clean

        test_data["H"] = np.concatenate(test_data["H"],axis=0)
        test_data["y"] = np.concatenate(test_data["y"],axis=0)
        test_data["x"] =  np.concatenate(test_data["x"],axis=0)
        x_tiled = np.tile(np.expand_dims(np.squeeze(test_data["x"]), axis=1), [1,test_data["H"].shape[1],1])
        y_clean = np.sum(x_tiled * test_data["H"], axis=-1,keepdims=True)
        test_data["y_clean"] = y_clean

        print("Finished!")

        print("Training data shape:")
        print(train_data["H"].shape)
        print("Validation data shape:")
        print(val_data["H"].shape)
        print("Test data shape:")
        print(test_data["H"].shape)

        if type(train_data["QAM_constellation"])==str:
            self.QAM_M = int(train_data["QAM_constellation"][-1].split("_")[-1])
        else:
            self.QAM_M = 4

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
          
        self.n_train = train_data["x"].shape[0]
        self.n_val = val_data["x"].shape[0]
        self.n_test = test_data["x"].shape[0]

    def reduce_test_set(self,n_files = 10240):
        self.test_data["H"] = self.test_data["H"][:n_files]
        self.test_data["x"] = self.test_data["x"][:n_files]
        self.test_data["y"] = self.test_data["y"][:n_files]
        self.test_data["y_clean"] = self.test_data["y_clean"][:n_files]
        self.n_test = n_files

    def get_noise_var(self,snr):
        noise_var = 1/np.sqrt(convert_from_decibel(snr))
        noise_var *= np.sqrt(self.n_sender/self.n_receiver) #channel matrix gain = Nt and noise equally distribuited among receivers
        return noise_var**2

    def generate_train(self):

        for H,y,x,y_clean in zip(self.train_data["H"],self.train_data["y"],self.train_data["x"],self.train_data["y_clean"]):
            #if np.random.uniform()>0.5:
            #    y = 0.5*y+0.5*np.expand_dims(y_clean,-1)
            noise_var = self.get_noise_var(self.db)
            if self.config["correlate"]:
                H = self.correlate_iid_channel(H,cR=self.cR,cT=self.cT)
                noise = y-y_clean
                y = H.dot(x)+noise
            yield {"H":H,"y":y,"x":x,"y_clean":y_clean,"noise_var":noise_var}

    def generate_val(self):

        for H,y,x,y_clean in zip(self.val_data["H"],self.val_data["y"],self.val_data["x"],self.val_data["y_clean"]):
            noise_var = self.get_noise_var(self.db)
            if self.config["correlate"]:
                H = self.correlate_iid_channel(H,cR=self.cR,cT=self.cT)
                noise = y-y_clean
                y = H.dot(x)+noise    
            yield {"H":H,"y":y,"x":x,"y_clean":y_clean,"noise_var":noise_var}

    def generate_test(self):

        for H,y,x,y_clean in zip(self.test_data["H"],self.test_data["y"],self.test_data["x"],self.test_data["y_clean"]):
            noise_var = self.get_noise_var(self.db)
            if self.config["correlate"]:
                H = self.correlate_iid_channel(H,cR=self.cR,cT=self.cT)
                noise = y-y_clean
                y = H.dot(x)+noise    
            yield {"H":H,"y":y,"x":x,"y_clean":y_clean,"noise_var":noise_var}
