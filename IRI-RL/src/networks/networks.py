import torch.nn as nn
import functools
from collections import OrderedDict
from torch.nn import init

class NetworksFactory:
	def __init__(self):
		pass


	@staticmethod
	def get_by_name(network_name, *args, **kwargs):
		if network_name == 'DQN_net':
			from .DQN_network import DQN_net
			network = DQN_net(*args, **kwargs)
		else:
			raise ValueError("Network %s not recognized." % network_name)

		return network


class NetworkBase(nn.Module):
	def __init__(self):
		super(NetworkBase, self).__init__()
		self._name = 'BaseNetwork'

	@property
	def name(self):
		return self._name

	def init_weights(self, net, init_type = 'normal', gain = 0.02):
		def init_func(m):
			classname = m.__class__.__name__
			if hasattr(m,'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
				if init_type == 'normal':
					init.normal_(m.weight.data, 0.0, gain)
				elif init_type == 'xavier':
					init.xavier_normal_(m.weight.data, gain = gain)
				elif init_type == 'kaiming':
					init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
				elif init_type == 'orthogonal':
					init.orthogonal_(m.weight.data, gain=gain)
				else:
					raise NotImplementedError('initialization method [%s] is not implmeented', init_type)
				if hasattr(m,'bias') and m.bias is not None:
					init.constant_(m.bias.data, 0.0)
			elif classname.find('BatchNorm2d') != -1:
				init.normal_(m.weight.data, 1.0, gain)
				init.constant_(m.bias.data, 0.0)
		net.apply(init_func)

	def _get_norm_layer(self, norm_type='batch'):
		if norm_type == 'batch':
			norm_layer = functools.partial(nn.BatchNorm2d, affine = True)
		elif norm_type == 'instance':
			norm_layer = functools.partial(nn.InstanceNorm2d, affine = True)
		elif norm_type == 'batchnorm2d':
			norm_layer = nn.BatchNorm2d
		else:
			raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

	# Question
	# Why not put a requires_grads = True case?
	def _set_requires_grads(self,net,requires_grads = True):
		if not requires_grads:
			for param in net.parameters():
				param.requires_grad = False

	# Question
	# Don't get what's the purpose of this function
	# Answer: get the ".module" out of the keys from the state_dict_checkpoint
	#		  must be some kind of format that stays when saved.
	def _clean_state_dict(self, state_dict_checkpoint):
		if 'module.' in list(state_dict_checkpoint.keys())[0]
			new_state_dict = OrderedDict()
			for k,v in state_dict_checkpoint.items():
				name = k[7:] #remove the ".module" part from the key
				new_state_dict[name] = v
		else:
			new_state_dict = state_dict_checkpoint
		return new_state_dict

	def print(self):
		num_params = 0
		for param in self.parameters():
			num_params += param.numel()
		print(self) # Question: what does this do?
		print('Total number of parameters: %d' % num_params)


