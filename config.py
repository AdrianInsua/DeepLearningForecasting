"""Global variables"""
STEPS = 3 			# Configure data window
LAYERS = 1			# Capas de la red
NEURONS = 256 		# Neurons to use in model
BATCH_SIZE = 32 	# Batch size to improve performance
NB_EPOCH = 1000	 	# Training epochs
LR = 0.0005			# Learning rate
TRSH = .7			# Clasificaton trheshold
SPLIT = .7			# data spliting percentage

""" User variables - also setted by command prompt """
PREPROCESS = True	# preprocess
GROUP = False		# grouping function
GROUP_BY = 'D'		# grouping criteria
SEASONALITY = False	# decompose seasonality
PRELOAD = True		# preload model
TRAIN = True		# train model
EVALUATE = True		# evaluate model
PREDICT = True		# make predictions
VERBOSE = 1			# verbose level
MODE = 'DEEP'		# type of model
MODEL = 'LSTM'		# model to use
FIELD = 'STK'	# field with relevant data
