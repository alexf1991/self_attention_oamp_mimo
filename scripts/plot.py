from absl import logging
from absl import app
from absl import flags
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tikzplotlib as tikz
from tensorboard.backend.event_processing import event_accumulator

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '../data/radar_data_proc_x_y_npy',
                    'Training data directory name')  # relative path from src dir
flags.DEFINE_string('model_dir', '', 'Save directory name')


def get_events_for_file(file_path):
    ea = event_accumulator.EventAccumulator(file_path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    data_dict = {}
    keys = ea.Tags()['tensors']
    # Create lists
    for key in keys:
        data_dict[key] = []
    # Get values
    for key in keys:
        tensor = ea.Tensors(key)
        for val in tensor:
            value = tf.io.decode_raw(val.tensor_proto.tensor_content, tf.float32).numpy()
            value = value[0]
            data_dict[key].append(value)

    return data_dict


def merge_dicts(dict_1, dict_2):
    for key in dict_1.keys():
        try:
            dict_1[key] += dict_2[key]
        except:
            print("Failed to merge : " + key)

    return dict_1


def get_all_events_for_mode(path, mode='test'):

    logs = os.listdir(os.path.join(path, mode))
    logs.sort(key=lambda x: os.path.getmtime(os.path.join(path, mode, x)))
    data_dict = None
    for l in logs:
        new_data_dict = get_events_for_file(os.path.join(path, mode, l))
        if data_dict != None:
            data_dict = merge_dicts(data_dict, new_data_dict)
        else:
            data_dict = new_data_dict

    return data_dict


def get_all_data(path):
    data_dict = {}
    data_dict["train"] = get_all_events_for_mode(path, 'train')
    data_dict["val"] = get_all_events_for_mode(path, 'val')
    data_dict["test"] = get_all_events_for_mode(path, 'test')
    return data_dict


def get_best_of_scalar(data_dict, name):
    val_data = data_dict["val"][name]
    test_data = data_dict["test"][name]
    if "f1_score" in name:
        best_idx = np.argwhere(val_data == np.max(val_data))[0][0]
    else:
        best_idx = np.argwhere(val_data == np.min(val_data)).ravel()
        best_idx_test = np.argwhere(test_data == np.min(test_data)).ravel()
        best_idx = best_idx[-1]

    best_val = test_data[best_idx]

    return best_idx, best_val


def get_gen_results(model_dir, metric="ser",include_models=[]):
    model_folders = glob.glob(os.path.join(model_dir, "*", ""))
    folder_dict = {}
    for mf in model_folders:
        snr_folders = glob.glob(os.path.join(mf,"all_snr", "*", ""))
        model_name = mf.split("/")
        if model_name[-1] == "":
            model_name = model_name[-2]
        else:
            model_name = model_name[-1]
        if not(model_name in include_models):
            continue

        if model_name not in folder_dict.keys():
            folder_dict[model_name] = {}

        for sf in snr_folders:
            key = sf.split("/")
            if key[-1] == "":
                key = key[-2]
            else:
                key = key[-1]

            correlations = os.listdir(sf)
            for corr in correlations:
                current_model_dir = os.path.join(sf,corr,"logs")
                data = get_all_events_for_mode(current_model_dir,"test")
                print(current_model_dir)
                print(data)
                y = data[metric]

                if key not in folder_dict[model_name].keys():
                    folder_dict[model_name][key] = {}

                folder_dict[model_name][key][corr] = y[-1]

    return folder_dict


def get_model_results(model_dir,metric="ser",include_models=[]):
    snr_folders = glob.glob(os.path.join(model_dir,"..","..","*", ""))
    #corr = model_dir.split("/")[-1]
    #print(model_dir)
    folder_dict = {}
    for snr in snr_folders:
        snr_val = snr.split("/")[-2]     
        
        for corr in range(0,10):
            corr =str(corr/10)

            path = os.path.join(snr,corr,"*")
            sub_models = glob.glob(path)
            for sf in sub_models:

                key = sf.split("/")
                if key[-1] == "":
                    key = key[-2]
                else:
                    key = key[-1]
                if not(key in include_models):
                    continue
                current_model_dir = os.path.join(sf,"logs")
                data = get_all_events_for_mode(current_model_dir,"test")
                if len(data.keys())==0:
                    continue

                y = data[metric]

                if not(key in folder_dict.keys()):
                    folder_dict[key]={}

                if not(snr_val in folder_dict[key].keys()):
                    folder_dict[key][snr_val] = {}

                folder_dict[key][snr_val][corr] = y[-1]

                
    return folder_dict

def get_folder(directory):
    all_files = os.listdir(directory)
    mmse_ser = 0.0
    ml_ser = 0.0
    bp_ser = 0.0
    try:
        # Try load baseline results
        for i_f, f in enumerate(all_files):
            data_mmse = np.load(os.path.join(directory, f.replace("channel", "results_mmse")))
            mmse_ser += np.mean(data_mmse["SE"])
            if os.path.exists(os.path.join(directory, f.replace("channel", "results_ml"))):
                data_ml = np.load(os.path.join(directory, f.replace("channel", "results_ml")))
                ml_ser += np.mean(data_ml["SE"])
                loaded_ml = True
            if os.path.exists(os.path.join(directory, f.replace("channel", "results_bp"))):
                data_bp = np.load(os.path.join(directory, f.replace("channel", "results_bp")))
                bp_ser += np.mean(data_bp["SE"])
                loaded_bp = True

        mmse_ser = mmse_ser / len(all_files)
        print("MMSE SER = " + str(mmse_ser))

        if loaded_bp:
            bp_ser = bp_ser / len(all_files)
            print("BP SER = " + str(bp_ser))
        if loaded_ml:
            ml_ser = ml_ser / len(all_files)
            print("ML SER = " + str(ml_ser))

        return mmse_ser

    except:
        print("No baseline results found")

def plot_over_snr(model_results_dict,corr,metric,models= ["transformer_model","oampnet2"]):

    plt.figure()
    all_results_train = {}
    all_db = {}
    legend = []
    for model in model_results_dict:
        snr_vals = model_results_dict[model].keys()
        print(model)
        print(snr_vals)
        for snr in snr_vals:
            data_trained = model_results_dict[model][snr][corr]
            db = int(snr.split("dB")[0])
            y_train = data_trained
            if not(np.isfinite(data_trained)):
                y = 1.0
            if model in all_results_train.keys():
                all_results_train[model].append(y_train)
                all_db[model].append(db)
            else:
                all_db[model] = [db]
                all_results_train[model]= [y_train]


    for model in models: 

        x = np.asarray(all_db[model])
        y_train = all_results_train[model]
        plt.plot(x, y_train, marker='*', linestyle='None')
        legend.append(model+" train")
        plt.yscale("log")
        plt.grid("on")
        #plt.ylim(0.5,10)
        plt.xlabel("SNR")
        plt.ylabel(metric)

    plt.legend(legend)
    plt.savefig("MIMO_snr_"+metric+".png", dpi=360)
    tikz.save("MIMO_snr_"+metric+".tikz")

def plot_over_corr(model_results_dict,snr,metric,models= ["transformer_model","oampnet2"]):

    plt.figure()
    all_results_train = {}
    all_corr = {}
    legend = []
    for model in model_results_dict:
        corrs = model_results_dict[model][snr].keys()
        for corr in corrs:
            if not(corr in model_results_dict[model][snr].keys()):
                continue
            data_trained = model_results_dict[model][snr][corr]
            corr = float(corr)
            y_train = data_trained
            if not(np.isfinite(data_trained)):
                y = 1.0
            if model in all_results_train.keys():
                all_results_train[model].append(y_train)
                all_corr[model].append(corr)
            else:
                all_corr[model] = [corr]
                all_results_train[model]= [y_train]


    for model in models: 

        x = np.asarray(all_corr[model])
        y_train = all_results_train[model]
        
        plt.plot(x, y_train, marker='*', linestyle='None')
        legend.append(model)

        plt.yscale("log")
        plt.grid("on")
        plt.xlabel("correlation")
        plt.ylabel(metric)


    plt.legend(legend)
    plt.savefig("MIMO_corr_"+metric+".png", dpi=360)
    tikz.save("MIMO_corr_"+metric+".tikz")
    
def main(argv):
    model_dir = FLAGS.model_dir
    data_dir = FLAGS.data_dir

    corr = model_dir.split("/")[-1]
    snr = model_dir.split("/")[-2]
    metric = "ser"
    models = ["oampsa","mmse","oampnet","oampnet2"]
    model_results_dict = get_model_results(model_dir,metric,models)
    print(model_results_dict)
    plot_over_snr(model_results_dict,corr,metric,models)
    plot_over_corr(model_results_dict,snr,metric,models)
    


if __name__ == "__main__":
    app.run(main)
