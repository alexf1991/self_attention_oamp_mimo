from absl import logging
from absl import app
from absl import flags
from utils.utils import *
from utils.trainer import ModelEnvironment
from utils.summary_utils import Summaries
from models.models import MMSE,OAMPNet,OAMPNet2,OAMPSA
from models.eval_functions.loss import EvalFunctions as EvalFunctions
import time
import numpy as np
import argparse
import json
import os
import sys
import tensorflow as tf
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def data_generator(feature_generator,batch_size,mode,
                   is_validation=False,
                   take_n=None,
                   skip_n=None,
                   shapes={}):
    if mode == "train":
        dataset = tf.data.Dataset.from_generator(feature_generator.generate_train,{"H":tf.float32,
                                                      "x":tf.float32,
                                                      "y":tf.float32,
                                                      "y_clean":tf.float32,
                                                      "noise_var":tf.float32})
    elif mode == "val":
        dataset = tf.data.Dataset.from_generator(feature_generator.generate_val,{"H":tf.float32,
                                                      "x":tf.float32,
                                                      "y":tf.float32,
                                                      "y_clean":tf.float32,
                                                      "noise_var":tf.float32})
    elif mode == "test":
        dataset = tf.data.Dataset.from_generator(feature_generator.generate_test,{"H":tf.float32,
                                                      "x":tf.float32,
                                                      "y":tf.float32,
                                                      "y_clean":tf.float32,
                                                      "noise_var":tf.float32})
    else:
        raise(ValueError)

    if mode =="train":
        dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(-1)
    else:
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(-1)

    return dataset


def learning_rate_fn(epoch):
    if epoch >= 20 and epoch < 30:
        return 0.01
    elif epoch >= 30 and epoch < 40:
        return 0.001
    elif epoch >= 40:
        return 0.001
    else:
        return 1.0


# ---------------------------------------------------------
# ---------------------------------------------------------

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '/tmp', 'save directory name')
flags.DEFINE_string('model_type', 'oampsa', 'Model type')
flags.DEFINE_string('mode', 'local', 'Mode for the training local or cluster')
flags.DEFINE_integer('start_epoch', 0, 'Number of epochs to train')
flags.DEFINE_integer('batch_size', 1024, 'Mini-batch size')
flags.DEFINE_integer('n_receiver', 16, 'Number of receive antennas')
flags.DEFINE_integer('n_sender', 8, 'Number of transmit antennas')
flags.DEFINE_integer('eval_every_n_th_epoch', 1,
                     'Integer describing after how many epochs the test and validation set are evaluted')
flags.DEFINE_string('config_file', '../config.json', 'Name of json configuration file')
flags.DEFINE_boolean('remove_old', False, "Remove old runs")


def main(argv):
    try:

        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

    except KeyError:

        task_id = 0

    model_save_dir = FLAGS.model_dir
    print("Saving model to : " + str(model_save_dir))
    start_epoch = FLAGS.start_epoch
    load_model = True
    batch_size = FLAGS.batch_size
    n_sender = FLAGS.n_sender
    n_receiver = FLAGS.n_receiver
    model_type = FLAGS.model_type
    remove_old = FLAGS.remove_old
    # load config file
    if FLAGS.config_file =='../config.json':
        file = os.path.abspath(__file__)
        file = file.split("/")[:-1]
        file = "/".join(file)
        config_file = os.path.join(file,FLAGS.config_file)
    try:
        print('*** loading config file: %s' % config_file)
        with open(config_file, 'r') as f:
            config = json.load(f)
            config["config_file_dir"] = config_file
            config["predictions_path"] = os.path.join(model_save_dir, "predictions")
            if not(os.path.exists(config["predictions_path"])):
                os.makedirs(config["predictions_path"])
    except:
        print('*** could not load config file: %s' % FLAGS.config_file)
        quit(0)

    # If load_model get old configuration
    if load_model:
        try:
            params = csv_to_dict(os.path.join(model_save_dir, "model_params.csv"))
        except:
            print("Could not find model hyperparameters!")


    path_to_data = config["data_path"].split("/")
    path_to_data = "/".join(path_to_data[:-1])
    
    data_list = os.listdir(path_to_data)
    
    input_shape_H = [batch_size, n_receiver*2,n_sender*2]
    input_shape_y = [batch_size, n_receiver*2,1]

    #Create training setttings for models
    H_data = tf.cast(tf.zeros(input_shape_H),tf.float32)
    y_data = tf.cast(tf.zeros(input_shape_y),tf.float32)
    noise_var = tf.cast(tf.ones([batch_size]),tf.float32)
    orig_model_save_dir = model_save_dir
    for i_c_str in range(1,4):
        model_save_dir = orig_model_save_dir
        ctcr_str = str(float(i_c_str) / 10.0)
        model_save_dir = model_save_dir.replace("YC",ctcr_str)
        for path in data_list:
            if "old" in path:
                continue

            print("Starting evaluation of dataset "+path)

            for i_c in range(1,4):
                ctcr = float(i_c)/10.0
                config["cR"] = ctcr
                config["cT"] = ctcr
                print("Evaluating correlation "+str(ctcr))
                if ctcr >0.0:
                    config["correlate"] = 1
                else:
                    config["correlate"] = 0

                config["data_path"] = os.path.join(path_to_data, path)

                data_loader = DataLoader(config)
                print(config)

                # Test data generator
                test_ds = data_generator(data_loader, batch_size, mode="test")
                n_test = data_loader.n_test

                # Create summaries to log
                scalar_summary_names = ['weight_decay_loss','total_loss','ser','ber','class_loss','wer','mhd','mde']

                if os.path.exists(os.path.join(model_save_dir, "all_snr", path,str(ctcr))):
                    if remove_old:
                        shutil.rmtree(os.path.join(model_save_dir, "all_snr", path,str(ctcr)))
                        print("Deleted " + os.path.join(model_save_dir, "all_snr", path,str(ctcr)))
                    else:
                        print("Skipped "+ctcr_str+" "+path+" "+str(ctcr))
                        continue

                # Create training settings for models
                M = int(np.sqrt(data_loader.QAM_M))
                sigConst = np.linspace(-M + 1, M - 1, M)
                sigConst /= np.sqrt((sigConst ** 2).mean())
                sigConst /= np.sqrt(2.)  # Each complex transmitted signal will have two parts
                QAM_constellation = tf.reshape(sigConst, shape=[M])

                if model_type == "oampsa":
                    model = OAMPSA(data_loader.n_sender, data_loader.n_receiver, config["iterations"],
                                              QAM_constellation=QAM_constellation)
                elif model_type == "oampnet":
                    model = OAMPNet(data_loader.n_sender, data_loader.n_receiver, config["iterations"])
                elif model_type == "oampnet2":
                    model = OAMPNet2(data_loader.n_sender, data_loader.n_receiver, config["iterations"])
                elif model_type == "MMSE":
                    model = MMSE(data_loader.n_sender, data_loader.n_receiver, config["iterations"])

                optimizer_type_fn = tf.keras.optimizers.Adam

                for x in test_ds:
                    x_init = [[x["H"], x["y"], x["noise_var"]]]
                    break

                summaries = Summaries(scalar_summary_names=scalar_summary_names,
                                      learning_rate_names=["learning_rate_classifier"],
                                      save_dir=os.path.join(model_save_dir, "all_snr", path,str(ctcr)),
                                      modes=["test"],
                                      summaries_to_print={"train": ["total_loss","ser"],
                                                          "test": ["total_loss", "ser","wer","mhd","mde"]})

                model_settings = [{'model': model,
                                   'optimizer_type': optimizer_type_fn,
                                   'base_learning_rate': 0.0,
                                   'learning_rate_fn': learning_rate_fn,
                                   'init_data': x_init,
                                   'load_model': True,
                                   'trainable': False}]
                # Build training environment
                env = ModelEnvironment(None,
                                       None,
                                       test_ds,
                                       0,
                                       EvalFunctions,
                                       feature_generator =data_loader,
                                       model_settings=model_settings,
                                       summaries=summaries,
                                       eval_every_n_th_epoch = 1,
                                       num_train_batches=1,
                                       load_model=load_model,
                                       save_dir = model_save_dir,
                                       loss_names=["total_loss"],
                                       input_keys=["H","y","noise_var"],
                                       label_keys=["x","y_clean"],
                                       start_epoch=start_epoch)

                results = env.predict_dataset(test_ds,num_batches = n_test//batch_size,use_progbar=True)



if __name__ == '__main__':
    app.run(main)

