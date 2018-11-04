
import argparse
def load_arguments():

    parser = argparse.ArgumentParser(description='RiboRL Model')
    parser.add_argument('--task', type= str, default='asite',
                        help='task name')
    parser.add_argument('--filename', type= str, default=None,
                        help='file name')
    parser.add_argument('--n_sample', type=int, default=1000000,
                        help='number of sample')
    parser.add_argument('--embsize', type=int, default=64,
                        help='size of word embeddings')
    parser.add_argument('--n_hids', type=int, default=192,
                        help='number of hidden units per layer')
    parser.add_argument('--n_filter', type=int, default=256,
                        help='number of kernel')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default= 1,
                        help='initial learning rate')
    parser.add_argument('--clip_grad', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=2000, metavar='N',
                        help='batch size')
    parser.add_argument('--model', type=int, default=15,
                        help='model type')
    parser.add_argument('--drate', type=float, default= 0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--prate', type=float, default= 0.02,
                        help='exploration applied to layers (0 = noexploration)')
    parser.add_argument('--load', type = str, default = None,
                        help='file path to load model')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--parallel', action='store_true',
                        help='use multi gpus')
    parser.add_argument('--interval', type=int, default=18, metavar='N',
                        help='report interval')
    parser.add_argument('--mark', type=str,  default='mark',
                        help='note to highlignt')
    parser.add_argument('--optim', type=str,  default='adadelta',
                        help='the name of optimizer,adadelta, sgd')
    parser.add_argument('--mode', type=int, default=0,
                        help='model mode, 0-train,1-eval,2-cv,3-looptest, 4-casestudy')
    parser.add_argument('--window', type=float, default=10,
                        help='default 10')
    parser.add_argument('--lambda1', type=float, default=0.0064,
                        help='default 1e-5')
    parser.add_argument('--L', type=float, default=0,
                        help='default fixed point')
    parser.add_argument('--thres', type=float, default=0,
                        help='default fixed point')
    parser.add_argument('--maskP', type=float, default=None,
                        help='None')
    parser.add_argument('--temp', type=int, default=0,
                        help='temp')
    args = parser.parse_args()

    return args

