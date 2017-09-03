import numpy as np
import tensorflow as tf

def fgm(model,eps=0.3,ord=np.inf,clip_min=None,clip_max=None,targeted=False,y=None):
    predictions = model.prob_predicts
    if y is None:
        y = tf.stop_gradient(tf.argmax(predictions,axis=-1)) #model.predicts
    else:
        y = tf.constant(y)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.logits, labels=y)
    if targeted:
        loss = -loss
    gradient = tf.gradients(loss,model.x_input)[0]

    if ord==np.inf:
        signed_grad = tf.sign(gradient)
    elif ord== 1:
        reduc_ind = list(range(1, len(model.x_input.get_shape())))
        signed_grad = gradient / tf.reduce_sum(tf.abs(gradient),
                                           reduction_indices=reduc_ind,
                                           keep_dims=True)
    elif ord == 2:
        reduc_ind = list(range(1,len(model.x_input.get_shape())))
        signed_grad = gradient / tf.sqrt(tf.reduce_sum(tf.square(gradient),
                                                       reduction_indices=reduc_ind,
                                                       keep_dims=True))
    else:
        raise NotImplementedError
    output = model.x_input + eps * signed_grad

    if (clip_min is not None) and (clip_max is not None):
        output = tf.clip_by_value(output, clip_min, clip_max)

    return output


def saliency_map(model):
    predictions = model.prob_predicts
    y = tf.stop_gradient(tf.argmax(predictions, axis=-1))
    logits = tf.gather_nd(model.logits,
                          tf.stack((tf.cast(tf.range(tf.shape(model.x_input)[0]),tf.int64),y),axis=1))
    gradients = tf.gradients(logits,model.x_input)[0]
    sm = tf.reduce_max(tf.abs(gradients),axis=-1)
    return sm
