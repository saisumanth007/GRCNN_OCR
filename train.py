import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
from model import *
from utils import *

TRAIN_SIZE = 1156
BATCH_SIZE = 100
NUM_BATCHES = TRAIN_SIZE // BATCH_SIZE
# NUM_BATCHES = 1
MAX_LEN = 26
NUM_EPOCHS = 5


tree = ET.parse("./data/word/word.xml")
root = tree.getroot()

if __name__ == "__main__":

	model = create_model()
	
	for e in xrange(NUM_EPOCHS):
		print "Epoch: "+str(e)
		for i in xrange(NUM_BATCHES):

			images = root[i:i+BATCH_SIZE]
			img_paths = [img.attrib['file'] for img in images]
			labels = [img.attrib['tag'] for img in images]
			lengths = [len(s) for s in labels]
			image_list=[]

			for img_path in img_paths:
				
				i_path = "./data/word/"+img_path
				img = cv.imread(i_path)
				# print i_path
				img = cv.resize(img,(100,32))
				img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
				img = np.reshape(img, (32,100,1))
				img = img.astype(float)
				img /= 255.0
				image_list.append(img)

			num_labels = str2label(labels,MAX_LEN)
			targets = np.asarray(num_labels)
			img_input = np.asarray(image_list)
			lengths = np.asarray(lengths)
			lengths = np.reshape(lengths, (BATCH_SIZE,1))
			lengths = lengths.astype(float)
			inp_lengths = np.ones((BATCH_SIZE,1)) * MAX_LEN
			inp_lengths = inp_lengths.astype(float)
			# inp_lengths = lengths
			
			inputs = {'the_input': img_input,
			          'the_labels': targets,
			          'input_length': inp_lengths,
			          'label_length': lengths
			          }
			outputs = {'ctc': np.zeros([BATCH_SIZE])}
			
			# print inputs.shape
			model.fit(inputs, outputs, batch_size = BATCH_SIZE)