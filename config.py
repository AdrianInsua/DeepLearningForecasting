"""Global variables"""

STEPS = 2 			# Configure data window
NEURONS = 128 		# Neurons to use in model
BATCH_SIZE = 128 	# Batch size to improve performance
NB_EPOCH = 200 		# Training epochs
LR = 0.005			# Learning rate
TRSH = .7			# Clasificaton trheshold
SPLIT = .8			# data spliting percentage

""" User variables - also setted by command prompt """
PREPROCESS = True	# preprocess
GROUP = True		# grouping function
GROUP_BY = 'M'		# grouping criteria
SEASONALITY = True	# decompose seasonality
PRELOAD = False		# preload model
TRAIN = True		# train model
PREDICT = True		# make predictions
VERBOSE = 1			# verbose level
MODE = 'LSTM'		# type of model
FIELD = 'IMPORTE'	# field with relevant data
