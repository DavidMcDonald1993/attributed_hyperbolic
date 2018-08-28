import numpy as np

import tensorflow as tf 
import keras.backend as K


# max_norm = np.nextafter(1, 0, dtype=K.floatx())
# max_ = np.finfo(K.floatx()).max
# min_norm = 1e-7


def minkowski_dot(x, y):
    # assert len(x.shape) == 2
    axes = len(x.shape) - 1, len(y.shape) -1
    return K.batch_dot( x[...,:-1], y[...,:-1], axes=axes) - K.batch_dot(x[...,-1:], y[...,-1:], axes=axes)
    # rank = x.shape[1] - 1
    # if len(y.shape) == 2:
    #     return K.sum(x[:,:rank] * y[:,:rank], axis=-1, keepdims=True) - x[:,rank:] * y[:,rank:]
    # else:
    #     return K.batch_dot( x[:,:rank], y[:,:,:rank], axes=[1,2]) - K.batch_dot(x[:,rank:], y[:,:,rank:], axes=[1, 2])


def hyperbolic_negative_sampling_loss(r, t):

    def loss(y_true, y_pred, r=r, t=t):

        r = K.cast(r, K.floatx())
        t = K.cast(t, K.floatx())

        u_emb = y_pred[:,0]
        samples_emb = y_pred[:,1:]
        
        inner_uv = minkowski_dot(u_emb, samples_emb)
        # inner_uv = K.clip(inner_uv, min_value=-np.inf, max_value=-(1+K.epsilon()))
        inner_uv = K.minimum(inner_uv, -(1+K.epsilon()))

        d_uv = tf.acosh(-inner_uv)
        out_uv = (K.square(r) - K.square(d_uv)) / t
        # out_uv = (r - d_uv) / t

        pos_out_uv = out_uv[:,0]
        neg_out_uv = out_uv[:,1:]
        
        pos_p_uv = tf.nn.sigmoid(pos_out_uv)
        neg_p_uv = 1. - tf.nn.sigmoid(neg_out_uv)

        pos_p_uv = K.clip(pos_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())
        neg_p_uv = K.clip(neg_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())

        return -K.mean(K.log(pos_p_uv) + K.sum(K.log(neg_p_uv), axis=-1))

    return loss

def hyperbolic_sigmoid_loss(y_true, y_pred,):

    u_emb = y_pred[:,0]
    samples_emb = y_pred[:,1:]
    
    inner_uv = minkowski_dot(u_emb, samples_emb)

    pos_inner_uv = inner_uv[:,0]
    neg_inner_uv = inner_uv[:,1:]
    
    pos_p_uv = tf.nn.sigmoid(pos_inner_uv)
    neg_p_uv = 1. - tf.nn.sigmoid(neg_inner_uv)

    pos_p_uv = K.clip(pos_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())
    neg_p_uv = K.clip(neg_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())

    return - K.mean( K.log( pos_p_uv ) + K.sum( K.log( neg_p_uv ), axis=-1) )

def euclidean_negative_sampling_loss(y_true, y_pred):

    u_emb = y_pred[:,0]
    samples_emb = y_pred[:,1:]
    
    inner_uv = K.batch_dot(u_emb, samples_emb, axes=[1,2])

    pos_inner_uv = inner_uv[:,0]
    neg_inner_uv = inner_uv[:,1:]
    
    pos_p_uv = tf.nn.sigmoid(pos_inner_uv)
    neg_p_uv = 1. - tf.nn.sigmoid(neg_inner_uv)

    pos_p_uv = K.clip(pos_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())
    neg_p_uv = K.clip(neg_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())

    return - K.mean( K.log( pos_p_uv ) + K.sum( K.log( neg_p_uv ), axis=-1) )


def hyperbolic_softmax_loss(y_true, y_pred,):

    u_emb = y_pred[:,0]
    samples_emb = y_pred[:,1:]
    
    inner_uv = minkowski_dot(u_emb, samples_emb)
    inner_uv = K.minimum(inner_uv, -(1+K.epsilon()))

    d_uv = tf.acosh(-inner_uv)
    exp_minus_d_uv_sq = K.exp(-K.square(d_uv))
    exp_minus_d_uv_sq = K.clip(exp_minus_d_uv_sq,  min_value=K.epsilon(), max_value=1-K.epsilon())
    return -K.mean(K.log(exp_minus_d_uv_sq[:,0]) - K.log(K.sum(exp_minus_d_uv_sq[:,1:], axis=-1)))

    # return - K.mean(K.log(tf.nn.softmax(inner_uv, axis=-1,)[:,0], ))
