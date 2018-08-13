"""Global variables"""

STEPS = 2 			# Configure data window
LAYERS = 2			# Capas de la red
NEURONS = 4 		# Neurons to use in model
BATCH_SIZE = 128 	# Batch size to improve performance
NB_EPOCH = 1000 		# Training epochs
LR = 0.0001			# Learning rate
TRSH = .7			# Clasificaton trheshold
SPLIT = .85			# data spliting percentage

""" User variables - also setted by command prompt """
PREPROCESS = True	# preprocess
GROUP = True		# grouping function
GROUP_BY = 'W'		# grouping criteria
SEASONALITY = True	# decompose seasonality
PRELOAD = False		# preload model
TRAIN = True		# train model
PREDICT = True		# make predictions
VERBOSE = 1			# verbose level
MODE = 'LSTM'		# type of model
FIELD = 'IMPORTE'	# field with relevant data
