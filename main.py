import numpy as np
import pandas as pd
import tensorflow as tf
import argparse,glob,os
from model import InceptionV3
from utils import load_images,show_image,save_image
from attack_method import fgm,saliency_map

from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser("InceptionV3 model.")
parser.add_argument('-c',"--checkpoint_file",default="./inception_v3.ckpt")
parser.add_argument('-b',"--batch_size",default=32,type=int)
args = parser.parse_args([])

batch_shape = [args.batch_size,299,299,3]
#image_true_labels = pd.read_csv("images/images.csv")
Id2name = pd.read_csv("images/categories.csv")
#image_lc = image_true_labels[['TrueLabel','ImageId']]\
#    .merge(image_true_cats,'left',left_on='TrueLabel',right_on='CategoryId')[['TrueLabel','CategoryName','ImageId']]
image_lc = pd.read_csv("images/images_all.csv")
image_iterator = load_images('images/images',batch_shape)


sess = tf.Session()
inceptionV3 = InceptionV3(num_classes=1001)
inceptionV3.restore(args.checkpoint_file,sess)

#adversarial generating
gen_adversarial_sample = fgm(inceptionV3,eps=0.01,clip_min=-1,clip_max=1)
sm = saliency_map(inceptionV3)

writer = tf.summary.FileWriter('./graphs',sess.graph)
writer.close()


predictedIds = []
adversarialPredictedIds = []
imageIds = []
for filenames, images in image_iterator:
    print ("current batch size:",len(filenames))
    adversarial_samples = sess.run(gen_adversarial_sample,feed_dict={inceptionV3.x_input:images})[:len(filenames)]
    predictions = sess.run([inceptionV3.predicts,sm],feed_dict={inceptionV3.x_input:images})
    adversarial_predictions = sess.run(inceptionV3.predicts,feed_dict={inceptionV3.x_input:adversarial_samples})[:len(filenames)]

    imageIds+=list(map(lambda x: x[:-4],filenames))
    predictedIds+=list(predictions[0][:len(filenames)])
    adversarialPredictedIds+=list(adversarial_predictions)

    for adv,fn in zip(adversarial_samples,filenames):
        save_image("output/Attack_"+fn,adv)

    # generate saliency maps

    #sms = sess.run(sm, feed_dict={inceptionV3.x_input: images})[:len(filenames)]
    #for sm_image,fn in zip(sms,filenames):
    #    save_image("output/sm_"+fn,(sm_image-1.0)*2.)


summary_frame = pd.DataFrame({"ImageId":imageIds,"predictedLabel":predictedIds,"adversarialLabel":adversarialPredictedIds})\
    .merge(image_lc,how='left',on='ImageId')
summary_frame = summary_frame.merge(Id2name,how='left',left_on='predictedLabel',right_on='CategoryId',suffixes=['_t','_p']).drop('CategoryId',axis=1)

print ("accuracy score:",accuracy_score(summary_frame.TrueLabel,summary_frame.predictedLabel))
print ("accuracy score under attack:",accuracy_score(summary_frame.TrueLabel,summary_frame.adversarialLabel))