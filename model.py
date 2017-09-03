import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

class InceptionV3(object):
    def __init__(self,num_classes):
        self.x_input = tf.placeholder(tf.float32,shape=[None,299,299,3],name="x_input")
        self.y = tf.placeholder(tf.int32,shape=[None],name="y_target")
        self.num_classes = num_classes

        #build inceptionV3
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            self.logits,self.end_points = inception.inception_v3(
                self.x_input,num_classes=num_classes,is_training=False)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.y,name='loss')
        self.prob_predicts = self.end_points['Predictions']
        self.predicts = tf.argmax(self.prob_predicts,axis=-1,name='predicts')

    def restore(self,checkpoint_file,sess):
        variable_to_restore = slim.get_model_variables(scope='InceptionV3')
        restore = slim.assign_from_checkpoint_fn(checkpoint_file, variable_to_restore)
        restore(sess)

    def predict(self,images,sess):
        return sess.run(self.prob_predicts,feed_dict={self.x_input:images})
