from absl import logging
from absl import app
from absl import flags
import time
import numpy as np
import argparse
import json
import os
import sys
import multiprocessing
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def learning_rate_fn(epoch):
    if epoch < 10:
        return 1.0
    elif epoch >= 10 and epoch < 20:
        return 0.1
    else:
        return 0.01


# ---------------------------------------------------------
# ---------------------------------------------------------

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '/tmp', 'save directory name')
flags.DEFINE_string('mode', 'local', 'Mode for the training local or cluster')
flags.DEFINE_string('model_type', 'oampsa', 'Model type')
flags.DEFINE_float('dropout_rate', 0.0, 'dropout rate for the dense blocks')
flags.DEFINE_float('weight_decay', 0.0, 'weight decay parameter')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('epochs', 25, 'number of epochs')
flags.DEFINE_integer('start_epoch', 0, 'Number of epochs to train')
flags.DEFINE_integer('batch_size',256, 'Mini-batch size')
flags.DEFINE_boolean('load_model', False, 'Bool indicating if the model should be loaded')
flags.DEFINE_integer('eval_every_n_th_epoch', 1,
                     'Integer describing after how many epochs the test and validation set are evaluted')
flags.DEFINE_string('config_file', '../config.json', 'Name of json configuration file')
flags.DEFINE_boolean('predict', False, "Is inference")
flags.DEFINE_boolean('use_softmax', False, "Use softmax output")
flags.DEFINE_boolean('remove_old', True, "Remove old runs")

def get_params(include_db=[],include_corr=[]):
    params=[]
    for i_db in range(5,17):
        if not(i_db in include_db):
            continue
        for i_c in range(0,10):
            if not(float(i_c/10) in include_corr):
                continue
            params.append([str(i_db)+"dB",float(i_c)/10.0])
    return params

def train_i(config,batch_size,model_save_dir,model_type,remove_old,eval_every_n_th_epoch,load_model,start_epoch,learning_rate,epochs):

    from utils.utils import DataLoader
    from utils.trainer import ModelEnvironment
    from utils.summary_utils import Summaries
    from models.models import MMSE, OAMPNet, OAMPNet2, OAMPSA
    from models.eval_functions.loss import EvalFunctions as EvalFunctions
    import tensorflow as tf

    def data_generator(feature_generator, batch_size, mode,
                       is_validation=False,
                       take_n=None,
                       skip_n=None,
                       shapes={}):
        if mode == "train":
            dataset = tf.data.Dataset.from_generator(feature_generator.generate_train, {"H": tf.float32,
                                                                                        "x": tf.float32,
                                                                                        "y": tf.float32,
                                                                                        "y_clean": tf.float32,
                                                                                        "noise_var": tf.float32})
        elif mode == "val":
            dataset = tf.data.Dataset.from_generator(feature_generator.generate_val, {"H": tf.float32,
                                                                                      "x": tf.float32,
                                                                                      "y": tf.float32,
                                                                                      "y_clean": tf.float32,
                                                                                      "noise_var": tf.float32})
        elif mode == "test":
            dataset = tf.data.Dataset.from_generator(feature_generator.generate_test, {"H": tf.float32,
                                                                                       "x": tf.float32,
                                                                                       "y": tf.float32,
                                                                                       "y_clean": tf.float32,
                                                                                       "noise_var": tf.float32})
        else:
            raise (ValueError)

        if mode == "train":
            dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(-1)
        else:
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(-1)

        return dataset


    data_loader = DataLoader(config)

    input_shape_H = [batch_size, data_loader.n_receiver * 2, data_loader.n_sender * 2]
    input_shape_y = [batch_size, data_loader.n_receiver * 2, 1]

    # Train data generator
    train_ds = data_generator(data_loader, batch_size, mode="train")
    # Train data generator
    val_ds = data_generator(data_loader, batch_size, mode="val")
    # Test data generator
    test_ds = data_generator(data_loader, batch_size, mode="test")
    # Create summaries to log
    scalar_summary_names = ['weight_decay_loss','total_loss','ser','ber','class_loss','wer','mhd','mde']

    if os.path.exists(os.path.join(model_save_dir, "logs")):
        if remove_old:
            shutil.rmtree(os.path.join(model_save_dir, "logs"))
            print("Deleted " + os.path.join(model_save_dir, "logs"))
        else:
            print("Skipped " + str(snr) + " " + " " + str(ctcr))
            return 0

    summaries = Summaries(scalar_summary_names=scalar_summary_names,
                          learning_rate_names=["learning_rate_classifier"],
                          save_dir=model_save_dir,
                          modes=["train", "val", "test"],
                          summaries_to_print={"train": ["total_loss", "ser"],
                                              "eval": ["total_loss", "ser"]})

    # Create training settings for models
    M = int(np.sqrt(data_loader.QAM_M))
    sigConst = np.linspace(-M + 1, M - 1, M)
    sigConst /= np.sqrt((sigConst ** 2).mean())
    sigConst /= np.sqrt(2.)  # Each complex transmitted signal will have two parts
    QAM_constellation = tf.reshape(sigConst, shape=[M])

    if model_type == "self_attention_model":
        model = OAMPSA(data_loader.n_sender, data_loader.n_receiver, config["iterations"],
                       QAM_constellation=QAM_constellation)
    elif model_type == "oampnet":
        model = OAMPNet(data_loader.n_sender, data_loader.n_receiver, config["iterations"])
    elif model_type == "oampnet2":
        model = OAMPNet2(data_loader.n_sender, data_loader.n_receiver, config["iterations"])
    elif model_type == "MMSE":
        model = MMSE(data_loader.n_sender, data_loader.n_receiver, config["iterations"])
    else:
        raise(NotImplementedError)

    optimizer_type_fn = tf.keras.optimizers.legacy.Adam

    for x in train_ds:
        x_init = [[x["H"], x["y"], x["noise_var"]]]
        break

    model_settings = [{'model': model,
                       'optimizer_type': optimizer_type_fn,
                       'base_learning_rate': learning_rate,
                       'learning_rate_fn': learning_rate_fn,
                       'init_data': x_init,
                       'load_model': False,
                       'trainable': True}]

    steps_test = data_loader.n_test // batch_size
    steps_train = data_loader.n_train // batch_size
    if model_type == "MMSE":
        steps_train = 1
        epochs = 1
        eval_every_n_th_epoch = 1

    # Build training environment
    trainer = ModelEnvironment(train_ds,
                               val_ds,
                               test_ds,
                               epochs,
                               EvalFunctions,
                               feature_generator=data_loader,
                               model_settings=model_settings,
                               summaries=summaries,
                               eval_every_n_th_epoch=eval_every_n_th_epoch,
                               num_train_batches=steps_train,
                               num_test_batches=steps_test,
                               load_model=load_model,
                               save_dir=model_save_dir,
                               loss_names=["total_loss"],
                               input_keys=["H", "y", "noise_var"],
                               label_keys=["x", "y_clean"],
                               start_epoch=start_epoch)

    trainer.train()

    return 1

def main(argv):
    try:

        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

    except KeyError:

        task_id = 0

    model_save_dir = FLAGS.model_dir
    print("Saving model to : " + str(model_save_dir))
    epochs = FLAGS.epochs
    start_epoch = FLAGS.start_epoch
    dropout_rate = FLAGS.dropout_rate
    weight_decay = FLAGS.weight_decay
    learning_rate = FLAGS.learning_rate
    load_model = FLAGS.load_model
    batch_size = FLAGS.batch_size
    model_type = FLAGS.model_type
    eval_every_n_th_epoch = FLAGS.eval_every_n_th_epoch
    remove_old = FLAGS.remove_old

    # Create parameter dict
    params = {}
    params["learning_rate"] = learning_rate
    params["model_dir"] = model_save_dir
    params["weight_decay"] = weight_decay
    params["dropout_rate"] = dropout_rate

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


    orig_config = config
    include_db = [5,6,7,8,9,10,11,12,13,14,15,16]
    include_corr = [0.0,0.1,0.2,0.3,0.4]
    params = get_params(include_db,include_corr)
    print(params)
    model_save_dir_orig = model_save_dir
    orig_data_path = config["data_path"]
    multiprocessing.set_start_method('spawn')
    for p in params:
        config = orig_config
        snr = p[0]
        ctcr = p[1]
        data_path = orig_data_path
        config["data_path"] = data_path.replace("XdB",snr)
        config["cT"]=ctcr
        config["cR"]=ctcr
        if ctcr <= 1e-6:
            config["correlate"] = 0
        else:
            config["correlate"] = 1

        model_save_dir = model_save_dir_orig
        model_save_dir = model_save_dir.replace("XdB",snr)
        model_save_dir = model_save_dir.replace("YC",str(ctcr))
        print("Saving to:")
        print(model_save_dir)
        print(config)

        p = multiprocessing.Process(target=train_i, args=(config,batch_size,model_save_dir,model_type,remove_old,eval_every_n_th_epoch,load_model,start_epoch,learning_rate,epochs))
        p.start()
        p.join()
        print("Finished training!")

if __name__ == '__main__':
    app.run(main)

