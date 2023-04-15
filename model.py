"""
define moduals of model
"""
from torch import flatten
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
	"""docstring for ClassName"""
	
	def __init__(self, args):
		super(CNNModel, self).__init__()
		##-----------------------------------------------------------
		## define the model architecture here
		## MNIST image input size batch * 28 * 28 (one input channel)
		##-----------------------------------------------------------

		conv_channel_out = args.channel_out2

		## define CNN layers below
		self.conv = nn.Sequential( 	nn.Conv2d(in_channels=1, out_channels=args.channel_out1, kernel_size=args.k_size, stride=args.stride),
									nn.ReLU(),
									nn.BatchNorm2d(args.channel_out1),
									nn.Dropout(args.dropout),
									nn.Conv2d(in_channels=args.channel_out1, out_channels=args.channel_out2, kernel_size=args.k_size, stride=args.stride),
									nn.ReLU(),
									nn.BatchNorm2d(args.channel_out2),
									nn.Dropout(args.dropout),
									nn.Conv2d(in_channels=args.channel_out2, out_channels=conv_channel_out, kernel_size=args.k_size, stride=args.stride),
									nn.ReLU(),
									nn.BatchNorm2d(conv_channel_out),
									nn.Dropout(args.dropout),
									
									nn.MaxPool2d(args.pooling_size, stride=args.max_stride)

								)

		##-------------------------------------------------
		## write code to define fully connected layers below
		##-------------------------------------------------
		in_size = conv_channel_out * 81
		out_size = 10
		# self.fc = nn.Linear(in_size, out_size)

		self.fc = nn.Sequential( nn.Linear(in_size, args.fc_hidden1),
								 nn.ReLU(),
								 nn.Linear(args.fc_hidden1, args.fc_hidden2),
								 nn.ReLU(),
								 nn.Linear(args.fc_hidden2, out_size)
								)
		

	'''feed features to the model'''
	def forward(self, x):  #default
		
		##---------------------------------------------------------
		## write code to feed input features to the CNN models defined above
		##---------------------------------------------------------
		x_out = self.conv(x)

		## write flatten tensor code below (it is done)
		x = flatten(x_out,1) # x_out is output of last layer
		

		## ---------------------------------------------------
		## write fully connected layer (Linear layer) below
		## ---------------------------------------------------
		result = self.fc(x)  # predict y
		
		
		return result
        
		
		
	
		