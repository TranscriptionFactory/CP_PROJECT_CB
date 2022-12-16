import os
import sys
import argparse

# current folder
cur_folder = 'batch_submit'

# storing paths as string
main_dir = os.getcwd()[:-len(cur_folder)]

# change dir


sys.path.append(main_dir + 'DehazeFormer/')

sys.path.append(main_dir + 'src/')

from my_models import *

data_hazy = str(main_dir + 'reside/hazy/')
data_clear = str(main_dir + 'reside/clear/')

dirs = {'hazy' : data_hazy, 'clear' : data_clear}


parser = argparse.ArgumentParser(description='usage: model type, output directory, epochs, learning rate')
parser.add_argument("--model",help='LightDehazeNet, LightDehazeNet_KL, LighDehazeNet_GL, LightDehazeNet_MI, LightDehaze_Net_Attn, LightDehazeNet_Attn_Conv, LightDehazeNet_Attn_Conv_Big')
parser.add_argument("--output_dir", help="output directory relative to main")
parser.add_argument("--epochs", default = "40")
parser.add_argument("--learning_rate", default = "0.02")

args = parser.parse_args()

output_weights = os.path.join(main_dir + 'weights/')

model_type = getattr(__import__('my_models'), args.model)#eval('args.model(output_weights + args.output_dir)')

created_model = model_type(output_weights + args.output_dir)
created_model.train(dirs['clear'], dirs['hazy'], args.epochs, args.learning_rate)