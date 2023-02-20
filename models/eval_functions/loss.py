import tensorflow as tf
import numpy as np


class EvalFunctions(object):
    """This class implements specialized operation used in the training framework"""
    def __init__(self,models,feature_generator):
        self.model = models[0]
        self.M = int(np.sqrt(feature_generator.QAM_M))
        sigConst = np.linspace(-self.M + 1, self.M - 1, self.M)
        sigConst /= np.sqrt((sigConst ** 2).mean())
        sigConst /= np.sqrt(2.)  # Each complex transmitted signal will have two parts
        self.QAM_constellation = tf.reshape(tf.cast(sigConst,tf.float32),shape=[1,1,self.M])
        tf.print(self.QAM_constellation)
        self.class_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def signal_to_symbol(self,signal,QAM_constellation):
        """ Takes the signal and outputs the corresponding symbols
            (no distinction between imag and real part here)
        Input:
            signal... signal-values
            QAM_constellation....all modulation scheme
        Output:
            symbols...symbol values (0,1,2,...)
        """
        signal = tf.expand_dims(signal,axis=-1)
        distance  = tf.abs(signal - QAM_constellation)
        symbols = tf.argmin(distance, axis=-1)
        return symbols

    def compute_ser(self,x,x_hat):
        """Computes the symbol error rate for all transmitte signals"""
        x_shp = tf.shape(x)
        x_real = x[:,:x_shp[-1]//2]
        x_imag = x[:,x_shp[-1]//2:]

        x_hat_real = x_hat[:,:x_shp[-1]//2]
        x_hat_imag = x_hat[:,x_shp[-1]//2:]
       
        L_real = x_real != x_hat_real
        L_imag = x_imag != x_hat_imag
       
        L = tf.logical_or(L_real,L_imag)
        L_shp = tf.shape(L)
        errors = tf.cast(L,tf.float32)
        nr_errors = tf.reduce_sum(errors)
        SER = nr_errors/tf.cast(L_shp[0]*L_shp[1],tf.float32)
    
        return SER

    def compute_wer(self,probs,labels):
        shp = tf.shape(probs)
        correct_probs = tf.where(labels==tf.reduce_max(labels,axis=-1,keepdims=True),probs,tf.zeros_like(probs))
        sum = tf.reduce_sum(correct_probs)
        mean = 1/tf.cast(shp[0],tf.float32)*sum
        wer = 1-mean
        return wer

    def compute_mhd(self,probs):
        shp = tf.shape(probs)
        correct_probs = tf.where(probs==tf.reduce_max(probs,axis=-1,keepdims=True),
                                 probs,
                                 tf.zeros_like(probs))
        sum = tf.reduce_sum(correct_probs)
        mean = 1/tf.cast(shp[0],tf.float32)*sum
        mhd = 1-mean
        return mhd

    def compute_mde(self,probs):
        shp = tf.shape(probs)
        sum = tf.reduce_sum(probs*tf.math.log(probs)/tf.math.log(2.0))
        mde = -1/tf.cast(shp[0],tf.float32)*sum
        return mde


    def compute_ser_prob(self,probs,labels):
        errors = tf.argmax(probs,axis=-1)!=tf.argmax(labels,axis=-1)
        errors = tf.cast(errors,tf.float32)
        nr_errors = tf.reduce_sum(errors)
        probs_shp = tf.shape(probs)
        SER = nr_errors/tf.cast(probs_shp[0]*probs_shp[1],tf.float32)
        
        return SER
	
    def compute_ber(self,x,x_hat):
        """Computes the bit error rate for all transmitte signals"""
        x_shp = tf.shape(x)
        errors = tf.cast(x!=x_hat,tf.float32)
        nr_errors = tf.reduce_sum(errors)
        BER = nr_errors/tf.cast(x_shp[0]*x_shp[1],tf.float32)
    
        return BER

    @tf.function
    def predict(self, inputs,training=True):
        """Returns a dict containing predictions e.g.{'predictions':predictions}"""
        x = self.model(inputs, training=training)
        return {'x':x}

    def to_real_imag_tensor(self, x):
        return tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)

    def to_complex_tensor(self, x):
        return tf.complex(x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:])

    def reset(self):
        return
    
        
    @tf.function
    def compute_loss(self, x, y, training=True,epoch=-1):
        """Has to at least return a dict containing the total loss and a prediction dict e.g.{'total_loss':total_loss},{'predictions':predictions}"""
        x_ests,probs = self.model(x, training=training)
        y_est = 0.0
        # Cross entropy losses
        x_target = tf.squeeze(y[0])
        y_target = tf.squeeze(x[1])
        y_clean =  tf.squeeze(y[1])

        x_est = x_ests[-1]
        shp = tf.shape(x_est)
        QAM_constellation = self.QAM_constellation
        QAM_constellation = tf.tile(QAM_constellation,[shp[0],shp[1],1])

        sym_est = self.signal_to_symbol(x_est,QAM_constellation)
        sym_target = self.signal_to_symbol(x_target,QAM_constellation)

        class_labels = tf.argmin(tf.abs(tf.expand_dims(x_target,-1)-self.QAM_constellation),axis=-1)
        class_labels = tf.one_hot(class_labels,self.M)
        class_shp = tf.shape(class_labels)

        class_loss = 0.0
        for i_pred,predictions in enumerate(probs):
            loss = self.class_loss(class_labels,predictions)
            class_loss += tf.reduce_mean(loss)

        if len(self.model.losses) > 0:
            weight_decay_loss = tf.add_n(self.model.losses)
        else:
            weight_decay_loss = 0.0

        total_loss = class_loss
        total_loss += weight_decay_loss
        
        ser = self.compute_ser(sym_est,sym_target)
        ber = self.compute_ber(sym_est,sym_target)

        if len(probs)>0:
            wer = self.compute_wer(probs[-1],class_labels)
            mde = self.compute_mde(probs[-1])
            mhd = self.compute_mhd(probs[-1])
        else:
            wer = 0.0
            mde = 0.0
            mhd = 0.0

        scalars = {'ber':ber,
                   'ser':ser,
                   'wer':wer,
                   'mhd':mhd,
                   'mde':mde,
                   'class_loss':class_loss,
                   'weight_decay_loss':weight_decay_loss,
                   'total_loss':total_loss}
        
        predictions = {'x':x_est}
        
        return scalars, predictions
    
    def post_train_step(self,args):
        return
