import numpy as np

import tensorflow as tf 
from tensorflow.python.framework import ops
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
        inner_uv = -inner_uv - 1.
        inner_uv = K.maximum(inner_uv, K.epsilon()) # clip to avoid nan

        d_uv = tf.acosh(1. + inner_uv) 
        d_uv_sq = K.square(d_uv)


        # r_sq = -K.stop_gradient(tf.nn.top_k(-K.flatten(d_uv_sq), k=512).values)[-1]
        # r = K.stop_gradient(K.mean(d_uv) - 2 * K.std( d_uv))
        # r_sq = K.stop_gradient(K.mean(d_uv) **2 )

        r_sq = K.square(r)


        out_uv = (r_sq - d_uv_sq) / t
        # out_uv = (K.square(r) - K.square(d_uv)) / t
        # out_uv = (r - d_uv) / t

        pos_out_uv = out_uv[:,0]
        neg_out_uv = out_uv[:,1:]
        
        pos_p_uv = tf.nn.sigmoid(pos_out_uv)
        neg_p_uv = 1. - tf.nn.sigmoid(neg_out_uv)

        pos_p_uv = K.clip(pos_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())
        neg_p_uv = K.clip(neg_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())

        return -K.mean(K.log(pos_p_uv) + K.sum(K.log(neg_p_uv), axis=-1))

    return loss

def hyperbolic_sigmoid_loss(y_true, y_pred):

    def hyperboloid_to_poincare_ball(X):
        return X[...,:-1] / (1. + X[...,-1:])

    # def euclidean_to_polar(X):
    #     r = K.sqrt(K.sum(X ** 2, axis=-1, keepdims=True))
    #     theta = tf.atan2(X[...,1:], X[...,:1])
    #     return K.concatenate([r, theta], axis=-1)

    def poincare_inner(a, b):

        a_r = K.sqrt(K.sum(K.square(a), axis=-1, keepdims=True))
        b_r = K.sqrt(K.sum(K.square(b), axis=-1, keepdims=False))
        return 4. * tf.atanh(a_r) * tf.atanh(b_r) * K.batch_dot(a, b, axes=(1, 2)) / (a_r * b_r)#* K.cos(a_theta - b_theta)

    # y_pred = hyperboloid_to_poincare_ball(y_pred)
    # y_pred = euclidean_to_polar(y_pred)

    u_emb = y_pred[:,0]
    samples_emb = y_pred[:,1:]

    # inner_uv = poincare_inner(u_emb, samples_emb)

    inner_uv = minkowski_dot(u_emb, samples_emb)
    inner_uv = -inner_uv - 1.
    inner_uv = K.maximum(inner_uv, K.epsilon()) # clip to avoid nan

    d_uv = tf.acosh(1. + inner_uv) 

    # d_uv_sq = K.square(d_uv)

    pos_d_uv = d_uv[:,0]
    neg_d_uv = d_uv[:,1:]

    sigma_sq = K.cast(9., dtype=K.floatx())
    # sigma_sq = K.maximum(sigma_sq, K.stop_gradient(K.mean(pos_d_uv)))
    # # sigma_sq = K.mean(d_uv)

    p = K.exp(-0.5 * K.square(pos_d_uv) / sigma_sq)
    # D = K.stop_gradient(K.exp(-0.5 * K.maximum(K.cast(10, dtype=K.floatx()), K.max(d_uv))**2 / sigma_sq))
    # q_lower = D / K.exp(-0.5 * K.square(neg_d_uv) / sigma_sq) 
    q_lower = K.exp(-0.5 * (K.stop_gradient(K.max(d_uv)**2) - neg_d_uv ** 2) / sigma_sq)
    q_upper = 1.


    pos_p_uv = p #/ (p + q_upper)
    neg_p_uv = q_lower #/ (q_lower + K.exp(-0.5 * K.square(neg_d_uv) / sigma_sq))

    # pos_log_p_uv = -0.5 * K.square(pos_d_uv) / sigma_sq
    # neg_log_p_uv = -0.5 * K.square(neg_d_uv) / sigma_sq

    # pos_p_uv = K.exp(pos_log_p_uv)
    # neg_p_uv = 1 - K.exp(neg_log_p_uv)

    pos_p_uv = K.clip(pos_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())
    neg_p_uv = K.clip(neg_p_uv, min_value=K.epsilon(), max_value=1-K.epsilon())

    return - K.mean( K.log( pos_p_uv ) + K.sum( K.log( neg_p_uv ), axis=-1) )
    # return - K.mean(pos_log_p_uv + K.sum(neg_log_p_uv + K.log(1. / neg_p_uv - 1.), axis=-1)) 

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
        exp_minus_d_uv_sq = K.maximum(exp_minus_d_uv_sq, K.epsilon()) 

        return -K.mean(K.log(exp_minus_d_uv_sq[:,0] / K.sum(exp_minus_d_uv_sq[:,0:], axis=-1)))

    return loss

def hyperbolic_softmax_loss(alpha=0):

    def np_arccosh_sq(x):
        return np.square(np.arccosh(x)).astype(K.floatx())
    
    def arccosh_sq_grad(op, grad):
        x = op.inputs[0]
        return grad * 2. * tf.acosh(x) / tf.sqrt(  tf.square(x) - 1.)

    def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name}): 
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

    def tf_arccosh_sq(x, name=None):

        with ops.op_scope([x], name, "arccosh_sq") as name:
            z = py_func(np_arccosh_sq,
                [x],
                [K.floatx()],
                name=name,
                grad=arccosh_sq_grad)  # <-- here's the call to the gradient
            return z[0]

    # def acosh(x):
    #     return K.log(x + K.sqrt(K.square(x) - 1))

    # def hyperboloid_to_poincare_ball(X):
    #     return X[...,:-1] / (1 + X[...,-1,None])

    def loss(y_true, y_pred, alpha=alpha):

        u_emb = y_pred[:,0]
        samples_emb = y_pred[:,1:]
        
        inner_uv = minkowski_dot(u_emb, samples_emb)
        inner_uv = -inner_uv - 1.
        inner_uv = K.maximum(inner_uv, K.epsilon()) # clip to avoid nan

        d_uv = tf.acosh(1. + inner_uv) 

        # d_uv /= K.stop_gradient(K.maximum(K.sqrt(K.mean(d_uv[:,0])), 1.))
        # mean = K.stop_gradient(K.mean(d_uv[:,0]))
        # std = K.stop_gradient(K.std(d_uv[:,0]))
        # sigma_sq = std **2
        sigma = K.cast(1., dtype=K.floatx())
        # sigma = K.maximum(sigma, K.mean(d_uv[:,0]))
        sigma_sq = K.stop_gradient(sigma ** 2)
        # sigma_sq = K.cast(0., dtype=K.floatx())
        # sigma_sq = K.maximum(sigma_sq, K.stop_gradient(K.mean(d_uv[:,0])))
        # d_uv /= K.sqrt(sigma_sq)
        minus_d_uv_sq = - 0.5 *  K.square(d_uv) / sigma_sq

        # exp_minus_d_uv_sq = K.exp(minus_d_uv_sq)
        # return -K.mean(K.log(exp_minus_d_uv_sq[:,0] / K.sum(exp_minus_d_uv_sq[:,1:], axis=-1)))

        # return -K.mean(minus_d_uv_sq[:,0]  - (max_ + K.log(K.sum(K.exp(minus_d_uv_sq[:,1:] - max_[:,None]),  axis=-1, keepdims=False))))
        # return -K.mean(minus_d_uv_sq[:,0] - tf.reduce_logsumexp(minus_d_uv_sq[:,1:],  axis=-1, keepdims=False))
        return  K.mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true[...,0], logits=minus_d_uv_sq))



    return loss
