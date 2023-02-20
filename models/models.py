import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class MMSE(tf.keras.Model):
    def __init__(self, n_sender=8, n_receiver=8, n_iter=10):
        super(MMSE, self).__init__()
        self.n_sender = n_sender
        self.n_receiver = n_receiver
        self.n_iter = n_iter
        self.ch_base = 32
        self.model_name = "MMSE"

    def build(self, input_shape):
        return

    def call(self, inputs, training):
        H = inputs[0]
        y = tf.squeeze(inputs[1])
        sigma2 = inputs[2]

        I = tf.eye(self.n_receiver * 2)
        I = tf.tile(tf.expand_dims(I, 0), [tf.shape(H)[0], 1, 1])
        sig = tf.reshape(sigma2, [-1, 1, 1]) * I / 2
        HH = tf.matmul(H, H, transpose_b=True)
        RR = tf.linalg.inv(HH + sig)
        H_inv = tf.matmul(H,RR,transpose_a=True)
        x = tf.matmul(H_inv,tf.expand_dims(y,-1))
        probs = []
        xs = [tf.squeeze(x)]

        return xs, probs



class OAMPWeightLayer(tf.keras.layers.Layer):
    def __init__(self,n_iter,name="",init="ones"):
        self.n_iter = n_iter
        self.weight_name = name
        self.init = init
        super(OAMPWeightLayer, self).__init__()

    def build(self, input_shape):
        self.layer_weights = self.add_weight(shape=[self.n_iter],
                                       initializer=self.init,
                                       name=self.weight_name,
                                       trainable=True)

    def call(self,iteration):
        return self.layer_weights[iteration]

class OAMPNet(tf.keras.Model):
    def __init__(self,n_sender, n_receiver, n_iter,ch_base=8, reg=1e-4,QAM_constellation=[-1.0 / np.sqrt(2.0),1.0 / np.sqrt(2.0)]):
        self.n_sender = n_sender
        self.n_receiver = n_receiver
        self.n_iter = n_iter
        self.reg = reg
        self.QAM_constellation = [tf.cast(const,tf.float32) for const in QAM_constellation]
        self.model_name = "OAMPNet"
        super(OAMPNet, self).__init__()

    def build(self,input_shape):
        self.damping = 0.1
        self.eps=1e-6
        self.beta=5e-10
        self.gamma = OAMPWeightLayer(self.n_iter,name="gamma",init="ones")
        self.theta = OAMPWeightLayer(self.n_iter,name="theta",init="zeros")

        return

    def shrink_bg_QPSK(self, r, rvar):

        p_tot = tf.cast(0.0, r.dtype)
        ps = []
        sp = tf.cast(0.0, r.dtype)
        const = tf.reshape(self.QAM_constellation,[1,1,len(self.QAM_constellation)])
        ps = self.normal_prob(const, r, rvar)
        p_tot = tf.reduce_sum(ps,axis=-1,keepdims=True)
        ps = ps/p_tot

        e_x = tf.reduce_sum(const*ps,axis=-1,keepdims=True)
        return e_x, ps

    def normal_prob(self, x, r, tau2):
        eps = tf.cast(5e-13, r.dtype)
        p = tf.maximum(tf.exp(-tf.square(x - r) / (2 * tau2)), eps)
        return p

    def update_v2(self,z,M,sigma2,HH):
        norm = tf.norm(z,axis=-2,keepdims=True)**2
        tr = tf.linalg.trace(HH)
        msig = tf.reshape(M * sigma2, [-1, 1, 1])
        v2 = (norm-msig)/tf.reshape(tr,[-1,1,1])
        v2 = tf.maximum(v2,self.eps)
        return v2

    def update_z(self,y,H,x):
        return y-tf.matmul(H, x)

    def update_r(self,x,W,z,iteration):
        return x+ self.gamma(iteration)*tf.matmul(W, z)

    def update_tau(self,v2,sigma2,trBB,trWW,N,iteration):
        return 1/(2*N)*(trBB*v2+self.theta(iteration)**2*trWW*sigma2)


    def update_W(self,v2,H,HH,I,sigma2,N):
        LMMSE = tf.linalg.inv(v2 * HH + tf.reshape(sigma2, [-1, 1, 1]) * I)
        W = v2*tf.matmul(H,LMMSE,transpose_a=True)
        norm = N/tf.linalg.trace(tf.matmul(W, H))
        return tf.reshape(norm,[-1,1,1])*W

    def oamp_iteration(self,x,y,H,HTH,HHT,sigma2,v2,W,IB,IW,trWW,trBB,iteration,N,M):
        z = self.update_z(y,H,x)
        r = self.update_r(x,W,z,iteration)
        v2 = self.update_v2(z,M,sigma2,HTH)
        tau2 = self.update_tau(v2,sigma2,trBB,trWW,N,iteration)
        x,p = self.shrink_bg_QPSK(r, tau2)
        W = self.update_W(v2,H,HHT,IW,sigma2,N)
        if iteration>self.n_iter-1:
            B = IB-self.theta(iteration+1)*tf.matmul(W,H)
            trWW = tf.linalg.trace(tf.matmul(W, W, transpose_b=True))
            trWW = tf.reshape(trWW,[-1,1,1])
            trBB = tf.linalg.trace(tf.matmul(B, B, transpose_b=True))
            trBB = tf.reshape(trBB,[-1,1,1])

        return x, v2, p,trWW,trBB,W

    def call(self, inputs, training):
        #Get inputs
        H, y, sigma2 = inputs
        sigma2 = tf.reshape(sigma2,[-1,1,1])/2
        H = tf.squeeze(H)
        #Get shape
        shp = tf.shape(H)
        batch_size = shp[0]
        M = shp[1]//2 #component wise complex-valued
        N = shp[2]//2
        #Precalc
        x = tf.zeros((batch_size, 2*N, 1), dtype=tf.float32)
        HTH = tf.matmul(H, H, transpose_a=True)
        HHT = tf.matmul(H, H, transpose_b=True)
        #Create identity matrix
        IB = tf.eye(2*N, batch_shape=[batch_size])
        IW = tf.eye(2*M, batch_shape=[batch_size])
        #Cast integers to floats
        N = tf.cast(N,dtype=tf.float32)
        M = tf.cast(M,dtype=tf.float32)
        #Calculate initial noise variance
        v2 = self.update_v2(y,M,sigma2,HTH)
        #Calculate initial inverse estimate
        W = self.update_W(v2,H,HHT,IW,sigma2,N)
        #Calculate initial traces
        B = IB-self.theta(0)*tf.matmul(W,H)
        trWW = tf.linalg.trace(tf.matmul(W, W, transpose_b=True))
        trWW = tf.reshape(trWW,[-1,1,1])
        trBB = tf.linalg.trace(tf.matmul(B, B, transpose_b=True))
        trBB = tf.reshape(trBB,[-1,1,1])
        #Initialize lists
        probs = []
        xs = []


        for iteration in range(self.n_iter):
            x, v2, p,trWW,trBB,W = self.oamp_iteration(x, y, H, HTH,HHT, sigma2, v2,W,IB,IW,trWW,trBB,iteration,N,M)
            xs.append(tf.squeeze(x))
            probs.append(p)

        return xs,probs

class OAMP(tf.keras.Model):
    def __init__(self, n_sender, n_receiver, n_iter, ch_base=8, reg=1e-4,
                 QAM_constellation=[-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]):
        self.n_sender = n_sender
        self.n_receiver = n_receiver
        self.n_iter = n_iter
        self.reg = reg
        self.QAM_constellation = [tf.cast(const, tf.float32) for const in QAM_constellation]
        self.model_name = "OAMP"
        super(OAMP, self).__init__()

    def build(self, input_shape):
        self.damping = 0.1
        self.eps = 1e-6
        self.beta = 5e-10

        return


    def shrink_bg_QPSK(self, r, rvar):

        p_tot = tf.cast(0.0, r.dtype)
        ps = []
        sp = tf.cast(0.0, r.dtype)
        const = tf.reshape(self.QAM_constellation,[1,1,len(self.QAM_constellation)])
        ps = self.normal_prob(const, r, rvar)
        p_tot = tf.reduce_sum(ps,axis=-1,keepdims=True)
        ps = ps/p_tot

        e_x = tf.reduce_sum(const*ps,axis=-1,keepdims=True)
        return e_x, ps

    def normal_prob(self, x, r, tau2):
        eps = tf.cast(5e-13, r.dtype)
        p = tf.maximum(tf.exp(-tf.square(x - r) / (2 * tau2)), eps)
        return p

    def update_v2(self,z,M,sigma2,HH):
        norm = tf.norm(z,axis=-2,keepdims=True)**2
        tr = tf.linalg.trace(HH)
        msig = tf.reshape(M * sigma2, [-1, 1, 1])
        v2 = (norm-msig)/tf.reshape(tr,[-1,1,1])
        v2 = tf.maximum(v2,self.eps)
        return v2

    def update_z(self,y,H,x):
        return y-tf.matmul(H, x)

    def update_r(self,x,W,z):
        return x+ tf.matmul(W, z)

    def update_tau(self,v2,sigma2,trBB,trWW,N):
        return 1/(2*N)*(trBB*v2+trWW*sigma2)


    def update_W(self,v2,H,HH,I,sigma2,N):
        LMMSE = tf.linalg.inv(v2 * HH + tf.reshape(sigma2, [-1, 1, 1]) * I)
        W = v2*tf.matmul(H,LMMSE,transpose_a=True)
        norm = N/tf.linalg.trace(tf.matmul(W, H))
        return tf.reshape(norm,[-1,1,1])*W

    def oamp_iteration(self,x,y,H,HTH,HHT,sigma2,v2,W,IB,IW,trWW,trBB,iteration,N,M):
        z = self.update_z(y,H,x)
        r = self.update_r(x,W,z)
        v2 = self.update_v2(z,M,sigma2,HTH)
        tau2 = self.update_tau(v2,sigma2,trBB,trWW,N)
        x,p = self.shrink_bg_QPSK(r, tau2)
        W = self.update_W(v2,H,HHT,IW,sigma2,N)
        B = IB-tf.matmul(W,H)
        trWW = tf.linalg.trace(tf.matmul(W, W, transpose_b=True))
        trWW = tf.reshape(trWW,[-1,1,1])
        trBB = tf.linalg.trace(tf.matmul(B, B, transpose_b=True))
        trBB = tf.reshape(trBB,[-1,1,1])

        return x, v2, p,trWW,trBB,W

    def call(self, inputs, training):
        #Get inputs
        H, y, sigma2 = inputs
        sigma2 = tf.reshape(sigma2,[-1,1,1])/2
        H = tf.squeeze(H)
        #Get shape
        shp = tf.shape(H)
        batch_size = shp[0]
        M = shp[1]//2 #component wise complex-valued
        N = shp[2]//2
        #Precalc
        x = tf.zeros((batch_size, 2*N, 1), dtype=tf.float32)
        HTH = tf.matmul(H, H, transpose_a=True)
        HHT = tf.matmul(H, H, transpose_b=True)
        #Create identity matrix
        IB = tf.eye(2*N, batch_shape=[batch_size])
        IW = tf.eye(2*M, batch_shape=[batch_size])
        #Cast integers to floats
        N = tf.cast(N,dtype=tf.float32)
        M = tf.cast(M,dtype=tf.float32)
        #Calculate initial noise variance
        v2 = self.update_v2(y,M,sigma2,HTH)
        #Calculate initial inverse estimate
        W = self.update_W(v2,H,HHT,IW,sigma2,N)
        #Calculate initial traces
        B = IB-tf.matmul(W,H)
        trWW = tf.linalg.trace(tf.matmul(W, W, transpose_b=True))
        trWW = tf.reshape(trWW,[-1,1,1])
        trBB = tf.linalg.trace(tf.matmul(B, B, transpose_b=True))
        trBB = tf.reshape(trBB,[-1,1,1])
        #Initialize lists
        probs = []
        xs = []


        for iteration in range(self.n_iter):
            x, v2, p,trWW,trBB,W = self.oamp_iteration(x, y, H, HTH,HHT, sigma2, v2,W,IB,IW,trWW,trBB,iteration,N,M)
            xs.append(tf.squeeze(x))
            v2s.append(v2)
            probs.append(p)

        return xs, probs


class OAMPNet2(tf.keras.Model):
    def __init__(self,n_sender, n_receiver, n_iter,ch_base=8, reg=1e-4,QAM_constellation=[-1.0 / np.sqrt(2.0),1.0 / np.sqrt(2.0)]):
        self.n_sender = n_sender
        self.n_receiver = n_receiver
        self.n_iter = n_iter
        self.reg = reg
        self.QAM_constellation = [tf.cast(const,tf.float32) for const in QAM_constellation]
        self.model_name = "OAMPNet2"
        super(OAMPNet2, self).__init__()


    def build(self, input_shape):
        self.damping = 0.1
        self.eps = 1e-6
        self.beta = 5e-10
        self.gamma = OAMPWeightLayer(self.n_iter,name="gamma",init="ones")
        self.theta = OAMPWeightLayer(self.n_iter,name="theta",init="ones")
        self.phi = OAMPWeightLayer(self.n_iter,name="phi",init="ones")
        self.zeta = OAMPWeightLayer(self.n_iter,name="zeta",init="zeros")


        return

    def shrink_bg_QPSK(self, r, rvar):

        p_tot = tf.cast(0.0, r.dtype)
        ps = []
        sp = tf.cast(0.0, r.dtype)
        const = tf.reshape(self.QAM_constellation,[1,1,len(self.QAM_constellation)])
        ps = self.normal_prob(const, r, rvar)
        p_tot = tf.reduce_sum(ps,axis=-1,keepdims=True)
        ps = ps/p_tot

        e_x = tf.reduce_sum(const*ps,axis=-1,keepdims=True)
        return e_x, ps

    def normal_prob(self, x, r, tau2):
        eps = tf.cast(5e-13, r.dtype)
        p = tf.maximum(tf.exp(-tf.square(x - r) / (2 * tau2)), eps)
        return p

    def update_v2(self,z,M,sigma2,HH):
        norm = tf.norm(z,axis=-2,keepdims=True)**2
        tr = tf.linalg.trace(HH)
        msig = tf.reshape(M * sigma2, [-1, 1, 1])
        v2 = (norm-msig)/tf.reshape(tr,[-1,1,1])
        v2 = tf.maximum(v2,self.eps)
        return v2

    def update_z(self,y,H,x):
        return y-tf.matmul(H, x)

    def update_r(self,x,W,z,iteration):
        return x+ self.gamma(iteration)*tf.matmul(W, z)

    def update_tau(self,v2,sigma2,trBB,trWW,N,iteration):
        return 1/(2*N)*(trBB*v2+self.theta(iteration)**2*trWW*sigma2)


    def update_W(self,v2,H,HH,I,sigma2,N):
        LMMSE = tf.linalg.inv(v2 * HH + tf.reshape(sigma2, [-1, 1, 1]) * I)
        W = v2*tf.matmul(H,LMMSE,transpose_a=True)
        norm = N/tf.linalg.trace(tf.matmul(W, H))
        return tf.reshape(norm,[-1,1,1])*W

    def oamp_iteration(self,x,y,H,HTH,HHT,sigma2,v2,W,IB,IW,trWW,trBB,iteration,N,M):
        z = self.update_z(y,H,x)
        r = self.update_r(x,W,z,iteration)
        v2 = self.update_v2(z,M,sigma2,HTH)
        tau2 = self.update_tau(v2,sigma2,trBB,trWW,N,iteration)
        x,p = self.shrink_bg_QPSK(r, tau2)
        W = self.update_W(v2,H,HHT,IW,sigma2,N)
        if iteration>self.n_iter-1:
            B = IB-self.theta(iteration+1)*tf.matmul(W,H)
            trWW = tf.linalg.trace(tf.matmul(W, W, transpose_b=True))
            trWW = tf.reshape(trWW,[-1,1,1])
            trBB = tf.linalg.trace(tf.matmul(B, B, transpose_b=True))
            trBB = tf.reshape(trBB,[-1,1,1])
        x = self.phi(iteration)*(x-self.zeta(iteration)*r)
        return x, v2, p,trWW,trBB,W

    def call(self, inputs, training):
        #Get inputs
        H, y, sigma2 = inputs
        sigma2 = tf.reshape(sigma2,[-1,1,1])/2
        H = tf.squeeze(H)
        #Get shape
        shp = tf.shape(H)
        batch_size = shp[0]
        M = shp[1]//2 #component wise complex-valued
        N = shp[2]//2
        #Precalc
        x = tf.zeros((batch_size, 2*N, 1), dtype=tf.float32)
        HTH = tf.matmul(H, H, transpose_a=True)
        HHT = tf.matmul(H, H, transpose_b=True)
        #Create identity matrix
        IB = tf.eye(2*N, batch_shape=[batch_size])
        IW = tf.eye(2*M, batch_shape=[batch_size])
        #Cast integers to floats
        N = tf.cast(N,dtype=tf.float32)
        M = tf.cast(M,dtype=tf.float32)
        #Calculate initial noise variance
        v2 = self.update_v2(y,M,sigma2,HTH)
        #Calculate initial inverse estimate
        W = self.update_W(v2,H,HHT,IW,sigma2,N)
        #Calculate initial traces
        B = IB-self.theta(0)*tf.matmul(W,H)
        trWW = tf.linalg.trace(tf.matmul(W, W, transpose_b=True))
        trWW = tf.reshape(trWW,[-1,1,1])
        trBB = tf.linalg.trace(tf.matmul(B, B, transpose_b=True))
        trBB = tf.reshape(trBB,[-1,1,1])
        #Initialize lists
        probs = []
        xs = []


        for iteration in range(self.n_iter):
            x, v2, p,trWW,trBB,W = self.oamp_iteration(x, y, H, HTH,HHT, sigma2, v2,W,IB,IW,trWW,trBB,iteration,N,M)
            xs.append(tf.squeeze(x))
            probs.append(p)

        return xs, probs


class MultiHeadedAttention(tf.keras.layers.Layer):

    def __init__(self, ch_base=8, attention_head_count=1, reg=1e-4):
        self.reg = reg
        self.ch_base = ch_base
        self.attention_head_count = attention_head_count
        super(MultiHeadedAttention, self).__init__()

    def build(self, input_shape):
        kernel_regularizer = tf.keras.regularizers.l2(self.reg)
        self.h_query = tf.keras.layers.Conv1D(self.ch_base * self.attention_head_count, kernel_size=[1],
                                     kernel_regularizer=kernel_regularizer,use_bias=False)
        self.h_key = tf.keras.layers.Conv1D(self.ch_base * self.attention_head_count, kernel_size=[1],
                                   kernel_regularizer=kernel_regularizer,use_bias=False)
        self.h_value = tf.keras.layers.Conv1D(self.ch_base * self.attention_head_count, kernel_size=[1],
                                     kernel_regularizer=kernel_regularizer,use_bias=False)
        self.h_ff = tf.keras.layers.Conv1D(input_shape[-1], kernel_size=[1], kernel_regularizer=kernel_regularizer,use_bias=False)

        self.scaled_dot_product = ScaledDotProductAttention(self.ch_base)

    def split_head(self, tensor, batch_size, d_h=None):
        if d_h == None:
            d_h = self.ch_base

        return tf.transpose(
            tf.reshape(
                tensor,
                (batch_size, -1, self.attention_head_count, d_h)
            ),
            [0, 2, 1, 3]
        )

    def concat_head(self, tensor, batch_size, d_h=None):
        if d_h == None:
            d_h = self.ch_base
        return tf.reshape(
            tf.transpose(tensor, [0, 2, 1, 3]),
            (batch_size, -1, self.attention_head_count * d_h))

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        x_query = self.h_query(inputs)
        x_key = self.h_key(inputs)
        x_value = self.h_value(inputs)

        x_query = self.split_head(x_query, batch_size)
        x_key = self.split_head(x_key, batch_size)
        x_value = self.split_head(x_value, batch_size)
        x_output, attention = self.scaled_dot_product(x_query, x_key, x_value)
        x_output = self.concat_head(x_output, batch_size)
        x_output = self.h_ff(x_output)

        return x_output


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, d_h):
        super(ScaledDotProductAttention, self).__init__()
        self.d_h = d_h

    def call(self, query, key, value, mask=None):
        matmul_q_and_transposed_k = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.d_h, dtype=tf.float32))
        scaled_attention_score = matmul_q_and_transposed_k / scale
        if mask is not None:
            scaled_attention_score += (mask * -1e9)
        attention_weight = tf.nn.softmax(scaled_attention_score, axis=-1)

        return tf.matmul(attention_weight, value), attention_weight


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, ch_base=8, attention_head_count=1, reg=1e-4):
        self.reg = reg
        self.ch_base = ch_base
        self.attention_head_count = attention_head_count
        super(EncoderLayer, self).__init__()

    def build(self, input_shape):
        kernel_regularizer = tf.keras.regularizers.l2(self.reg)
        self.h_ff_1 = tf.keras.layers.Conv1D(input_shape[-1], kernel_size=[1], kernel_regularizer=kernel_regularizer,use_bias=False)
        self.h_ff_2 = tf.keras.layers.Conv1D(input_shape[-1], kernel_size=[1], kernel_regularizer=kernel_regularizer,use_bias=False)
        self.multi_headed_attention = MultiHeadedAttention(self.ch_base,
                                                                  attention_head_count=self.attention_head_count,
                                                                  reg=self.reg)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training):
        x_output = self.multi_headed_attention(inputs, training)
        x_output += inputs
        x_output = self.layer_norm_1(x_output)
        sc = x_output

        x_output = self.h_ff_1(x_output)
        x_output = tf.nn.relu(x_output)
        x_output = self.h_ff_2(x_output)
        x_output += sc
        x_output = self.layer_norm_2(x_output)
        return x_output

class SALayerMIMO(tf.keras.layers.Layer):

    def __init__(self, n_sender, n_receiver, n_iter, ch_base=8, reg=1e-4,constellation=[]):
        self.n_sender = n_sender
        self.n_receiver = n_receiver
        self.n_iter = n_iter
        self.reg = reg
        self.ch_base = ch_base
        self.constellation = tf.cast(constellation,tf.float32)
        self.attention_head_count = 1
        super(SALayerMIMO, self).__init__()

    def build(self, input_shape):
        kernel_regularizer = tf.keras.regularizers.l2(self.reg)

        self.encoder_layer = EncoderLayer(ch_base=self.ch_base, attention_head_count=self.attention_head_count,
                                                 reg=self.reg)
        self.conv_h = tf.keras.layers.Conv1D(1, kernel_size=1, padding="same", kernel_regularizer=kernel_regularizer,
                                    use_bias=False,name="h_conv")
        self.conv_p0 = tf.keras.layers.Conv1D(self.ch_base, kernel_size=1, kernel_regularizer=kernel_regularizer,
                                     padding="same",use_bias=False)
        self.conv_p = tf.keras.layers.Conv1D(len(self.constellation), kernel_size=1, kernel_regularizer=kernel_regularizer, padding="same",name="p_conv",use_bias=False)
        self.eps = 1e-6
        self.norm = tf.keras.layers.LayerNormalization(epsilon=self.eps)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=self.eps)


    def call(self, y, sigma2, H,x_last, v2, S,p_last,r,iteration, training):

        shp = tf.shape(H)
        batch_size = shp[0]

        y_tensor = tf.tile(tf.reshape(y,[batch_size,1,-1]),[1,shp[-1],1])
        y_tensor = y_tensor/(tf.sqrt(tf.reduce_sum(y_tensor**2,axis=-1,keepdims=True))+self.eps)
        x_last_tensor = tf.reshape(x_last, [batch_size,-1, 1])
        H_tensor = H/(tf.sqrt(tf.reduce_sum(H ** 2, axis=-2, keepdims=True)) +self.eps)
        p_last_tensor = tf.reshape(p_last,[batch_size,p_last.shape[1],p_last.shape[-1]])


        r_tensor = tf.reshape(r, [batch_size,-1, 1])
        v2_tensor = tf.tile(tf.reshape(tf.sqrt(v2),[batch_size,1,1]),[1,shp[-1],1])

        in_tensor = tf.concat(
            [S,
             tf.transpose(H_tensor,perm=[0,2,1]),
             y_tensor,
             x_last_tensor,
             p_last_tensor,
             r_tensor,
             v2_tensor
             ], axis=-1)

        in_tensor = self.norm(in_tensor)

        in_tensor_ = tf.reshape(in_tensor, [batch_size, -1, in_tensor.shape[-1]])

        x_output = self.encoder_layer(in_tensor_, training)
        if training:
            x_output = tf.nn.dropout(x_output,rate=0.2)
        S = x_output[..., :self.ch_base]
        x_dense = x_output
        xh = self.conv_h(tf.reduce_mean(x_output,axis=1,keepdims=True))

        v2 = tf.nn.sigmoid(xh)**2
        x_dense = self.conv_p0(x_dense)
        x_dense = self.norm_2(x_dense)
        x_dense = tf.nn.relu(x_dense)
        p_x = self.conv_p(x_dense)
        p_x = tf.nn.sigmoid(p_x)
        N_c = len(self.constellation)
        p_train = p_x

        return p_train,p_x,v2, S

class OAMPSA(tf.keras.Model):
    def __init__(self,n_sender, n_receiver, n_iter,ch_base=16, reg=1e-4,QAM_constellation=[-1.0 / np.sqrt(2.0),1.0 / np.sqrt(2.0)]):
        self.n_sender = n_sender
        self.n_receiver = n_receiver
        self.n_iter = n_iter
        self.reg = reg
        self.QAM_constellation = [tf.cast(const,tf.float32) for const in QAM_constellation]
        self.model_name = "OAMPSA"
        self.ch_base = ch_base
        super(OAMPSA, self).__init__()


    def build(self, input_shape):
        self.damping = 0.1
        self.eps = 1e-6
        self.beta = 5e-10

        self.self_attention_layer = SALayerMIMO(self.n_sender, self.n_receiver, self.n_iter,
                                                ch_base=self.ch_base,
                                                reg=self.reg,
                                                constellation=self.QAM_constellation)
        return

    def calculate_expectation(self, p_x,QAM_constellation):

        const = tf.reshape(QAM_constellation,[1,1,-1])
        p_tot = tf.reduce_sum(p_x,axis=-1,keepdims=True)
        p = p_x/p_tot

        e_x = tf.reduce_sum(const*p,axis=-1,keepdims=True)
        return e_x

    def update_v2(self,z,M,sigma2,HH):
        norm = tf.norm(z,axis=-2,keepdims=True)**2
        tr = tf.linalg.trace(HH)
        msig = tf.reshape(M * sigma2, [-1, 1, 1])
        v2 = (norm-msig)/tf.reshape(tr,[-1,1,1])
        v2 = tf.maximum(v2,self.eps)
        return v2

    def update_z(self,y,H,x):
        return y-tf.matmul(H, x)

    def update_r(self,x,W,z):
        return x+ tf.matmul(W, z)

    def update_W(self,v2,H,HH,I,sigma2,N):
        LMMSE = tf.linalg.inv(v2 * HH + tf.reshape(sigma2, [-1, 1, 1]) * I)
        W = v2*tf.matmul(H,LMMSE,transpose_a=True)
        norm = N/tf.linalg.trace(tf.matmul(W, H))
        return tf.reshape(norm,[-1,1,1])*W

    def oamp_iteration(self,x,y,p,H,HTH,HHT,sigma2,v2,W,IB,IW,trWW,trBB,S_old,iteration,N,M,training):
        z = self.update_z(y,H,x)
        r = self.update_r(x,W,z)
        v2 = self.update_v2(z,M,sigma2,HTH)
        p,p_x,v2, S_new = self.self_attention_layer(y, sigma2, H, x, v2, S_old, p, r,iteration, training)

        N_c = len(self.QAM_constellation)
        const = tf.reshape(self.QAM_constellation,[N_c])
        x = self.calculate_expectation(p_x,const)
        W = self.update_W(v2,H,HHT,IW,sigma2,N)
        if iteration>self.n_iter-1:
            B = IB-self.theta(iteration+1)*tf.matmul(W,H)
            trWW = tf.linalg.trace(tf.matmul(W, W, transpose_b=True))
            trWW = tf.reshape(trWW,[-1,1,1])
            trBB = tf.linalg.trace(tf.matmul(B, B, transpose_b=True))
            trBB = tf.reshape(trBB,[-1,1,1])

        return x, v2, p,p_x,trWW,trBB,W,S_new

    def call(self, inputs, training):
        #Get inputs
        H, y, sigma2 = inputs
        sigma2 = tf.reshape(sigma2,[-1,1,1])/2
        H = tf.squeeze(H)
        #Get shape
        shp = tf.shape(H)
        batch_size = shp[0]
        M = shp[1]//2 #component wise complex-valued
        N = shp[2]//2
        #Precalc
        x = tf.zeros((batch_size, 2*N, 1), dtype=tf.float32)
        HTH = tf.matmul(H, H, transpose_a=True)
        HHT = tf.matmul(H, H, transpose_b=True)
        #Create identity matrix
        IB = tf.eye(2*N, batch_shape=[batch_size])
        IW = tf.eye(2*M, batch_shape=[batch_size])
        #Cast integers to floats
        N = tf.cast(N,dtype=tf.float32)
        M = tf.cast(M,dtype=tf.float32)
        #Calculate initial noise variance
        v2 = self.update_v2(y,M,sigma2,HTH)
        #Calculate initial inverse estimate
        W = self.update_W(v2,H,HHT,IW,sigma2,N)
        #Calculate initial traces
        B = IB-tf.matmul(W,H)
        trWW = tf.linalg.trace(tf.matmul(W, W, transpose_b=True))
        trWW = tf.reshape(trWW,[-1,1,1])
        trBB = tf.linalg.trace(tf.matmul(B, B, transpose_b=True))
        trBB = tf.reshape(trBB,[-1,1,1])
        #Initialize lists
        probs = []
        xs = []
        #Initialize state
        BS = tf.shape(y)[0]
        S = tf.zeros([BS,2*self.n_sender, self.ch_base])
        p_x = tf.tile(tf.reshape(tf.zeros([len(self.QAM_constellation)]), [1, 1, len(self.QAM_constellation)]),
                      [batch_size, self.n_sender * 2, 1])
        for iteration in range(self.n_iter):
            x, v2, p,p_x,trWW,trBB,W,S = self.oamp_iteration(x, y,p_x, H, HTH,HHT, sigma2, v2,W,IB,IW,trWW,trBB,S,iteration,N,M,training)
            xs.append(tf.squeeze(x,axis=-1))
            probs.append(p)

        return xs, probs
