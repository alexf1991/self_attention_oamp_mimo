import sys
import os
from absl import logging
from absl import app
from absl import flags
import numpy as np


class Channel:
    # This class contains various methods to generate the channel matrix
    # a) Gaussian Channel
    # b) Kronecker Channel
    # and to generate simultated transmissions

    def __init__(self, parameters, QAM_constellation):
        self.constellation = parameters['constellation']
        self.snr = parameters['snr_db']
        self.channel_type = parameters['channel_type']
        self.nr_rx = parameters['nr_rx']
        self.nr_tx = parameters['nr_tx']
        self.normalized = parameters['normalized']
        self.QAM_values = QAM_constellation

        if self.channel_type == "GAUSS":
            self.H = self.get_gauss_iid()
        elif self.channel_type == "KRONECKER":
            # H = self.get_kronecker()
            print("not implemented")

    ## -------------------------------------------------------
    def get_gauss_iid(self):
        """ Computes the Gaussian Channel Matrix """
        if self.normalized is True:
            H_real = np.random.normal(size=(self.nr_rx, self.nr_tx), scale=1. / np.sqrt(2. * self.nr_rx))
            H_imag = np.random.normal(size=(self.nr_rx, self.nr_tx), scale=1. / np.sqrt(2. * self.nr_rx))
        else:
            H_real = np.random.normal(size=(self.nr_rx, self.nr_tx), scale=1)
            H_imag = np.random.normal(size=(self.nr_rx, self.nr_tx), scale=1)

        # concatenate along nr_tx (Re(H), -Im(H))
        h1 = np.concatenate((H_real, -1. * H_imag), axis=1)

        # concatenate along nr_tx (Im(H), Re(H))
        h2 = np.concatenate((H_imag, H_real), axis=1)

        # Put together full channel-matrix (see (2.7) in Masters Thesis);
        # now we have: (Re(H), -Im(H); Im(H), Re(H))
        H = np.concatenate((h1, h2), axis=0)
        if self.normalized is True:
            # normalize every column of H which is the same as normalizing the
            # complex values itself if we sum up Re(1)^2+Im(1)^2+... anyway
            # h_i = { h_{ij} / \sqrt{ \sum h_{ij}^2  } }
            H = H / np.sqrt(np.sum(H ** 2, axis=0))

        return H

    ## -------------------------------------------------------
    def get_channel_matrix(self):
        return self.H

    ## -------------------------------------------------------
    def get_random_x(self):

        # draw 2 times nr_transmitter QAM values at random.
        # we always have a combination of 2 values to represent the actual complex-valued signal
        # x = np.random.choice(QAM_values, size=2*parameters['nr_tx'])
        x = np.random.choice(self.QAM_values, size=2 * self.nr_tx)
        x.shape = (x.shape[0], 1)
        return x

    ## -------------------------------------------------------
    def get_noise_sigma(self):
        noise_sigma = np.ones((1, 1)) * 1 / np.sqrt(10 ** (self.snr / 10))
        noise_sigma = noise_sigma * np.sqrt(self.nr_tx / self.nr_rx)
        return noise_sigma

    ## -------------------------------------------------------
    def get_noise_variance(self):
        return self.get_noise_sigma() ** 2

    ## -------------------------------------------------------
    def get_noise_std(self):
        return self.get_noise_sigma()

    ## -------------------------------------------------------
    def transmission(self, x):
        """ Simulates one transmission
        Input:
            x...codeword to be transmitted
        Output:
            y...transmitted codeword over H with noise#
        """

        # draw noise-terms with unit std.-dev
        noise_real = np.random.normal(size=(1, self.nr_rx), scale=1 / np.sqrt(2.))
        noise_imag = np.random.normal(size=(1, self.nr_rx), scale=1 / np.sqrt(2.))
        noise = np.concatenate((noise_real, noise_imag), axis=1)

        # compute noise std.-dev for specified snr
        noise_sigma = np.ones((1, 1)) * 1 / np.sqrt(10 ** (self.snr / 10))
        noise_sigma = np.repeat(noise_sigma, 2 * self.nr_rx, axis=1)
        # take channel gain into account
        noise_sigma = noise_sigma * np.sqrt(self.nr_tx / self.nr_rx)

        # rescale noise terms accordingly
        noise = noise * noise_sigma
        y = np.matmul(self.H, x) + noise.T

        return y

## -------------------------------------------------------
def generate_QAM(nr_symbols):
    """ Computes the values for the QAM lattice (only in 1D)
    Input:
        nr_symbols... QAM order
    Output:
        QAM_values... QAM lattice values
    """
    # pdb.set_trace()
    axis_min = int(-np.sqrt(nr_symbols)+1)
    axis_max = int(np.sqrt(nr_symbols)-1)
    QAM_values = np.linspace(axis_min, axis_max, int(np.sqrt(nr_symbols)))
    # compute the avg. power and normalize
    normalization = np.sqrt( np.mean(QAM_values**2))*np.sqrt(2)
    QAM_values = QAM_values/(normalization)
    return QAM_values


## -------------------------------------------------------
def specify_channel_parameters(**kwargs):
    QAM_order = kwargs.get("QAM_order", 4)
    parameters = {'iid': kwargs.get("iid", True),
                  'normalized': kwargs.get("normalized", True),
                  'nr_tx': kwargs.get("nr_tx", 8),
                  'nr_rx': kwargs.get("nr_rx", 16),
                  'snr_db': kwargs.get("snr_db", 20),
                  'signal_strength': kwargs.get("signal_strength", 1),
                  'channel_type': kwargs.get("channel_type", "GAUSS")
                  }

    # normally use nr_tx = 16 and QAM = 16
    parameters["constellation"] = generate_QAM(QAM_order)
    return parameters


# ---------------------------------------------------------
# ---------------------------------------------------------

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp', 'save directory name')

def main(argv):
    cluster = False

    # catch experiment_id (if given)
    if len(sys.argv) == 2:
        setting_id = sys.argv[1]
        print("SETTING_ID=" + setting_id)
    else:
        setting_id = 999
        print("no setting id given")

    ## define experimental settings
    # Signal to Noise Ratio (in DB)
    snr_range = range(5,15) #5,6,..14
    # Restricted to QAM now (bc. of SBP)
    QAM_order = 4
    # Number of random instances
    nr_runs = 2000
    # Number of transmitter and receiver (8/8, 8/16,...)
    nr_tx = 8
    nr_rx = 16

    # define all path variables
    dir_results = FLAGS.data_dir

    # get the SLURM ID
    if cluster is True:
        print("SLURM_ARRAY_TASK_ID=" + os.environ['SLURM_ARRAY_TASK_ID'] + "\n")
        run_id = np.uint16(os.environ['SLURM_ARRAY_TASK_ID'])  # Slurm ID
        loops = range(1)
    else:
        run_id = 0
        loops = range(100)


    for snr in snr_range:
        for loop in loops:
            if len(loops)>1:
                run_id = loop

            print("Processing SNR of : "+str(snr))
            ex_id = '_' + str(setting_id).zfill(3) + '_' + str(run_id).zfill(4)

            channel_settings = os.path.join(os.path.join(dir_results,str(snr)+"dB"),'channel' + ex_id + '.npz')

            if not(os.path.exists(os.path.join(dir_results,str(snr)+"dB"))):
                os.makedirs(os.path.join(dir_results,str(snr)+"dB"))
            ## define arrays for saving the experimental results
            H_collect = np.zeros((nr_runs, 2 * nr_rx, 2 * nr_tx))
            x_collect = np.zeros((nr_runs, 2 * nr_tx, 1))
            y_collect = np.zeros((nr_runs, 2 * nr_rx, 1))

            ## specify the paramters of the system
            parameters = specify_channel_parameters(QAM_order=QAM_order, nr_tx=nr_tx, nr_rx=nr_rx, snr_db=snr)
            QAM_constellation = generate_QAM(QAM_order)
            parameters["QAM_order"] = QAM_order

            ## perform nr_runs different channels and decode the signals
            for run in range(nr_runs):
                # create a random channel according to our parameters

                channel = Channel(parameters, QAM_constellation)

                # compute specific realization and store it
                H = channel.get_channel_matrix()  # get channel matrix and store it
                assert H.shape == H_collect[run, :, :].shape
                H_collect[run, :, :] = H

                random_x = channel.get_random_x()  # get random signal and store it
                assert random_x.shape == x_collect[run, :].shape
                x_collect[run, :, :] = random_x

                y = channel.transmission(random_x)  # compute transmitted signal and store it
                assert y.shape == y_collect[run, :].shape
                y_collect[run, :, :] = y

            print("-------")
            print('saving outputs...')
            print(channel_settings)
            # safe channel settings
            np.savez(channel_settings,
                     H=H_collect,
                     y=y_collect,
                     x=x_collect,
                     QAM_constellation=QAM_constellation,
                     QAM_order=QAM_order,
                     nr_rx=nr_rx,
                     nr_tx=nr_tx,
                     snr=snr)

    print('-end-')


if __name__ == '__main__':
    app.run(main)

