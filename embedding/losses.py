import numpy as np

import tensorflow as tf 
import keras.backend as K

def minkowski_dot(x, y):
    axes = len(x.shape) - 1, len(y.shape) -1
    return K.batch_dot(x[...,:-1], y[...,:-1], axes=axes) - K.batch_dot(x[...,-1:], y[...,-1:], axes=axes)

def hyperbolic_negative_sampling_loss(r, t):

    def loss(y_true, y_pred, r=r, t=t):

        r = K.cast(r, K.floatx())
        t = K.cast(t, K.floatx())

        u_emb = y_pred[:,0]
        samples_emb = y_pred[:,1:]
        
        inner_uv = minkowski_dot(u_emb, samples_emb)
        inner_uv = K.minimum(inner_uv, -(1+K.epsilon()))

        d_uv = tf.acosh(-inner_uv)
        # d_uv_sq = K.square(d_uv)


        r = -K.stop_gradient(tf.nn.top_k(-d_uv, k=1).values)
        # r = K.stop_gradient(K.mean(d_uv) - 2 * K.std( d_uv))
        # r_sq = K.stop_gradient(K.mean(d_uv_sq) - 1 * K.std( d_uv_sq))

        # out_uv = (r_sq - d_uv_sq) / t
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

def euclidean_softmax_loss(alpha=0):

    def loss(y_true, y_pred, alpha=alpha):

        alpha = K.cast(alpha, K.floatx())

        u_emb = y_pred[:,0]
        samples_emb = y_pred[:,1:]

        d_uv = K.sqrt(K.sum(K.square(u_emb - samples_emb)))

        exp_minus_d_uv_sq = K.exp(-d_uv)
        exp_minus_d_uv_sq = K.maximum(exp_minus_d_uv_sq, K.cast(1e-45, K.floatx())) 

        return -K.mean(K.log(exp_minus_d_uv_sq[:,0] / K.sum(exp_minus_d_uv_sq[:,0:], axis=-1)))

    return loss

def hyperbolic_softmax_loss(alpha=0):

    def acosh(x):
        return K.log(x + K.sqrt(K.square(x) - 1))

    def hyperboloid_to_poincare_ball(X):
        return X[...,:-1] / (1 + X[...,-1,None])

    def scale(x):
        return tf.nn.sigmoid((x - K.stop_gradient(K.mean(x, axis=-1, keepdims=True))) /\
                 K.stop_gradient(K.std(x, axis=-1, keepdims=True)))

    def loss(y_true, y_pred, alpha=alpha):

        alpha = K.cast(1e-3, K.floatx())

        u_emb = y_pred[:,0]
        samples_emb = y_pred[:,1:]
        
        inner_uv = minkowski_dot(u_emb, samples_emb)
        inner_uv = K.minimum(inner_uv, -(1+K.epsilon())) # clip to avoid nan

        # minus_d_uv = -tf.acosh(-inner_uv)
        minus_d_uv = -K.square(tf.acosh(-inner_uv) )
        # minus_d_uv = scale(minus_d_uv)
        logits = minus_d_uv

        # u_emb_poincare = hyperboloid_to_poincare_ball(u_emb)
        # samples_emb_poincare = hyperboloid_to_poincare_ball(samples_emb)

        # u_rank = K.sqrt(K.sum(K.square(u_emb_poincare), axis=-1, keepdims=True))
        # samples_rank = K.sqrt(K.sum(K.square(samples_emb_poincare), axis=-1, keepdims=False))

        # u_rank = u_emb[...,-1,None]
        # samples_rank = samples_emb[...,-1]

        # d_uv_sq = K.square(tf.acosh(-inner_uv))
        # logits = -(1. + alpha * (samples_rank - u_rank)) * d_uv_sq
        logits -= K.stop_gradient(K.max(logits, axis=-1, keepdims=True, ))

        return K.mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true[...,0], logits=logits))
        # return K.mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true[...,0], logits=-d_uv_sq))
        # return K.mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true[...,0], logits=logits))

        

    return loss

def hyperbolic_hinge_loss(alpha=0):
# def hyperbolic_softmax_loss(alpha=0):


    def acosh(x):
        return K.log(x + K.sqrt(K.square(x) - 1))

    def hyperboloid_to_poincare_ball(X):
        return X[...,:-1] / (1 + X[...,-1,None])

    def scale(x):
        return tf.nn.sigmoid((x - K.stop_gradient(K.mean(x, axis=-1, keepdims=True))) /\
                 K.stop_gradient(K.std(x, axis=-1, keepdims=True)))

    def loss(y_true, y_pred, alpha=alpha):

        alpha = K.cast(1e-3, K.floatx())

        u_emb = y_pred[:,0]
        samples_emb = y_pred[:,1:]
        
        inner_uv = minkowski_dot(u_emb, samples_emb)
        inner_uv = K.minimum(inner_uv, -(1+K.epsilon())) # clip to avoid nan

        # minus_d_uv = -tf.acosh(-inner_uv)
        d_uv = K.square(tf.acosh(-inner_uv) ) 

        return -K.mean( inner_uv[:,0] - K.log(K.sum(K.exp(inner_uv[:,1:]), axis=-1, keepdims=True)))

    return loss



