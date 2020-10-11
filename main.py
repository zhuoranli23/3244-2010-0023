import os
from argparse import ArgumentParser
import model
from utils import create_link


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--load_height', type=int, default=64) # changed from 286
    parser.add_argument('--load_width', type=int, default=64) # changed from 286
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=64) # changed from 256
    parser.add_argument('--crop_width', type=int, default=64) # changed from 256
    parser.add_argument('--test_crop_height', type=int, default=128)  # changed from 256
    parser.add_argument('--test_crop_width', type=int, default=128)  # changed from 256
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--idt_coef', type=float, default=0.5)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/sketch2pokemon')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/sketch2pokemon')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='resnet_9blocks')
    parser.add_argument('--dis_net', type=str, default='n_layers')
    parser.add_argument('--test_length', type=int, default=8, help="number of testing data")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    create_link(args.dataset_dir)

    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    
    md = model.cycleGAN(args)
    md.train(args)


if __name__ == '__main__':
    main()
