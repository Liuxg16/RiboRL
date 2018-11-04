# coding=utf-8
# python2
import  sys, time, os , random, math,logging
import cPickle as pickle, json
import numpy as np
import scipy.stats
from collections import Counter
from os.path import dirname, join
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from networks import *
from dataio import Dataset_yeast
import  options

def main(args,kwargs):

    kwargs['model'] = args.model
    kwargs['prate'] = args.prate
    kwargs['drate'] = args.drate
    kwargs['n_hids'] = args.n_hids
    kwargs['n_filter'] = args.n_filter
    kwargs['embsize'] = args.embsize
    kwargs['mark'] = args.mark
    kwargs['optim'] = args.optim
    kwargs['window'] = args.window
    kwargs['task'] = args.task
    kwargs['lambda1'] = args.lambda1
    kwargs['parallel'] = args.parallel
    kwargs['seed'] = args.seed
    kwargs['L'] = args.L
    kwargs['n_sample'] = args.n_sample
    kwargs['mask_position'] = args.maskP
    rnd_seed = args.seed
    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)

    dataset = Dataset_yeast(1234, 1.0/3, relative_offset=args.window)
    test_data,test_labels = dataset.getData()
    vocab = dataset.vocab

    kwargs['n_tokens'] = len(vocab)
    tmodel = TrainingModel(model_name= RiboRL, types ='rationales',vocab = vocab,  **kwargs)

    if args.load is  not None: 
        with open(args.load, 'rb') as f:
            tmodel.model.load_state_dict(torch.load(f))
    test_data = tmodel.gen_batches([test_data,test_labels])
    tmodel.validation(test_data)
    test_loss, test_mse, test_reward ,corr= tmodel.validation(test_data)
    print('| Testing|loss {:5.3f} | reward {:5.3} | mse {:5.5}| corr {:5.5f}|'.\
                format(test_loss, test_reward,test_mse,corr))

class TrainingModel(object):

    def __init__(self, model_name,types,vocab, **kwargs):
        self.args = args
        self.lr = args.lr
        self.eval = False
        self.kwargs = kwargs
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.best_val_reward = None
        self.device_ids = kwargs['device_ids']
        ###############################################################################
        # Build the model
        ###############################################################################
        n_labels = 2
        self.model = model_name(types, n_labels,  **kwargs)
        self.model_pre = self.model
        if self.kwargs['parallel'] :
            self.model = nn.DataParallel(self.model_pre, self.device_ids)
        else:
            self.model = self.model_pre

        if args.cuda:
            self.model.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.kwargs['optim']=='sgd' :
            self.optimizer = optim.SGD(parameters,lr = self.lr)
        elif self.kwargs['optim']=='adadelta' :
            self.optimizer = optim.Adadelta(parameters,lr = self.lr)
        elif self.kwargs['optim']=='adam' :
            self.optimizer = optim.Adam(parameters,lr = self.lr)
        elif self.kwargs['optim']=='sgd-mom' :
            self.optimizer = optim.SGD(parameters,lr = self.lr, momentum=0.9)

    def validation(self, data_source):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.0
        total_perf = 0
        acc = 0
        edge_acc = 0
        last_edge_acc = 0
        reward = 0
        n_batches = len(data_source[0])
        num_samples = 0
        preds = []
        labels = []
        for i in range(n_batches):
            data, targets = self.get_batch(data_source, i, torch.FloatTensor, evaluation=True)
            batch_size = data.size()[0]
            loss,rewardd, square,  output = self.model(data, targets)
            reg_loss = torch.sum(loss)
            accuracy_ = torch.mean(square)
            reward_ = torch.mean(rewardd)

            acc +=        accuracy_.data[0] * batch_size
            reward +=     reward_.data[0]* batch_size
            total_loss += reg_loss.data[0] * batch_size
            num_samples += batch_size

            pred = output[0]
            pr = pred.data.cpu().numpy()
            preds.append(pr)
            tar = targets.view(-1,1).data.cpu().numpy()
            labels.append(tar)
            # for p,l in zip(pr.tolist(), tar.tolist()):
            #     print p,l
        
        Y_pred = np.vstack(preds).reshape(-1)
        Y = np.vstack(labels).reshape(-1)
        # print loss
        # print np.sum(Y_pred)
        # print np.sum(Y.shape)

        # # R^2 value for our predictions on the training set
        pearsonr =  scipy.stats.pearsonr(Y.flatten(),
                                   Y_pred.flatten())[0]

        return total_loss /num_samples,acc/num_samples, reward/num_samples, pearsonr

    def get_batch(self, source, i, warper_tensor = torch.LongTensor,evaluation=False, warp= True):
        data_ts  =  torch.LongTensor(source[0][i])
        target_ts = warper_tensor(source[1][i])
        if self.args.cuda:
            data_ts = data_ts.cuda()
            target_ts = target_ts.cuda()
        if warp:
            data = Variable(data_ts, volatile=evaluation)
            target = Variable(target_ts)
            return data, target
        else:
            return data_ts, target_ts

    def gen_batches(self,data_tuple):
        # suitable for abitrary length of data_tuple
        batches = [[] for i in xrange(len(data_tuple))]
        for i in xrange(len(data_tuple)-1):
            assert len(data_tuple[i]) == len(data_tuple[i+1])
        bs = self.args.batch_size
        for i in xrange(int(np.ceil(len(data_tuple[0]) / float(bs)))):
        # for i in xrange(int(np.floor(len(data_tuple[0]) / float(bs)))):  # delete the last batch
            for j in xrange(len(data_tuple)):
                batches[j].append(data_tuple[j][i * bs:i * bs + bs])
        return batches

    def create_buckets_batches(self,lstx,lsty,mask_id=0):
        
        assert min(len(x) for x in lstx) > 0
        batches_x, batches_y = [], []
        assert len(lstx) == len(lsty)
        bs = self.args.batch_size
         
        for i in xrange(int(np.ceil(len(lstx) / float(bs)))):
            bucket_list_x = lstx[i*bs:i*bs+bs]            
            mx_num_sent = max(len(x) for x in bucket_list_x) +3
            bucket_docs = [[mask_id]*3+ doc + [mask_id] * (mx_num_sent - len(doc)) for doc in bucket_list_x]
            # bucket_docs = [doc + [mask_id] * (mx_num_sent - len(doc)) for doc in bucket_list_x]
            bucket_docs_np = np.array(bucket_docs)
            batches_x.append(bucket_docs_np)
            bucket_list_y = np.array(lsty[i*bs:i*bs+bs])
            batches_y.append(bucket_list_y)
        return batches_x,batches_y

    def gen_idx2word(self, vocab, id):
        idx2word = dict((word,idx) for idx,word in vocab.items())
        if idx2word.has_key(id):
            return idx2word[id]
        else:
            return 'unk'

class mylog(object):

    def __init__(self,dirname):
        filename = dirname+'/logging.log'
        logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=filename,
                        filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-4s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def logging(self,str):
        logging.info("\n"+str)

if __name__ == "__main__":

    kwargs = {
        "max_epochs":10000,
        "n_repeat":10,
        "filter_list":[6,8,10],
        "device_ids":[0,1,2,3],
        "MAX_NORM":3,
        'clip_grad':0.25

    }
    
    args = options.load_arguments()
    main(args,kwargs)




