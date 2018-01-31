import keras
from keras.models import Model
from keras.layers import *
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras import backend as K
import numpy as np

#Config
nclasses = 36
nl = nclasses+1
max_len = 26
train_batch_size = 10
val_batch_size = 10
epochs = 1

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def create_model():

	def GRCL(inp, n_out, n_iter, f_size):
	
		conv_rec = Conv2D(n_out, (f_size, f_size), padding = 'same')

		conv_gate_rec = Conv2D(n_out, (1,1))

		for i in xrange(n_iter):

			if i==0:

				#Feed forward
				conv_f = Conv2D(n_out, (f_size, f_size), padding = 'same')(inp)
				bn_f = BatchNormalization()(conv_f)
				x = Activation('relu')(bn_f)

				#Gated
				conv_gate_f = Conv2D(n_out, (f_size, f_size), padding = 'same')(inp)
				bn_gate_f = BatchNormalization()(conv_gate_f)

			else:

				c_rec = conv_rec(x)
				bn_rec = BatchNormalization()(c_rec)

				c_gate_rec = conv_gate_rec(x)
				bn_gate_rec = BatchNormalization()(c_gate_rec)
				gate_add = Add()([bn_gate_rec, bn_gate_f])
				gate = Activation('sigmoid')(gate_add)

				gate_mul = Multiply()([bn_rec,gate])
				bn_gate_mul = BatchNormalization()(gate_mul)
				x_add = Add()([bn_f, bn_gate_mul])

				x = Activation('relu')(x_add)

		return x

	# Model

	#GRCNN Block

	inp = Input(name = "the_input", shape = (32, 100, 1))
	conv1 = Conv2D(64, (3,3), padding = 'same')(inp)
	act1 = Activation('relu')(conv1)
	pool1 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(act1)
	grcl1 = GRCL(pool1, 64, 5, 3)
	pool2 = MaxPooling2D(pool_size = (2,2), strides = (2,2))(grcl1)
	grcl2 = GRCL(pool2, 128, 5, 3)
	zero_pad3 = ZeroPadding2D(padding=(0,1))(grcl2)
	pool3 = MaxPooling2D(pool_size = (2,2), strides = (2,1))(zero_pad3)
	grcl3 = GRCL(pool3, 256, 5, 3)
	zero_pad4 = ZeroPadding2D(padding=(0,1))(grcl3)
	pool4 = MaxPooling2D(pool_size = (2,2), strides = (2,1))(zero_pad4)
	conv_final = Conv2D(512, (2,2))(pool4)
	bn_final = BatchNormalization()(conv_final)
	out = Activation('relu')(bn_final)

	rout= Reshape((int(out.shape[1]*out.shape[2]),int(out.shape[3])))(out)
	lstm1=Bidirectional(LSTM(512, return_sequences=True),merge_mode='sum')(rout)
	lstm2=Bidirectional(LSTM(nclasses+1,return_sequences=True),merge_mode='sum')(lstm1)
	y_pred = Activation('softmax')(lstm2)

	labels = Input(name='the_labels', shape=[max_len], dtype='float32')
	input_length = Input(name='input_length', shape=[1], dtype='float32')
	label_length = Input(name='label_length', shape=[1], dtype='float32')

	loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

	# sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
	ada = Adadelta(lr=0.01, rho=0.9)

	model = Model(inputs=[inp, labels, input_length, label_length], outputs=loss_out)
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

	return model