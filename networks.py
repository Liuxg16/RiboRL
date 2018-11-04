import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PolicyNet(nn.Module):
    '''
    add feedback
    '''
    def __init__(self, input_size, hidden_size, cate_num, length):
        super(PolicyNet, self).__init__()
        self.hidden_size = hidden_size
        self.cate_num = cate_num
        self.length = length+1

        self.gen = nn.GRUCell(input_size+self.length, hidden_size)
        self.cf = nn.Linear(hidden_size, cate_num) # regression layer 

    def forward(self, inputs):
        '''
        inputs: (b_s,l,emb_size)
        return: (b_s,l)
        '''
        inputs_p = inputs
        b_s = inputs_p.size()[0]
        steps = inputs_p.size(1)
        outputs = Variable(torch.zeros(b_s,steps).cuda())
        pis = Variable(torch.zeros(b_s,steps).cuda())
        ones = Variable(torch.ones(b_s,1).cuda())

        self.prate = 0.02
        self.switch_m = Variable(torch.FloatTensor(b_s, 2).fill_(self.prate)).cuda()
        self.switch_m[:,1] = 1- self.prate
        self.action_m = Variable(torch.FloatTensor(b_s,self.cate_num).fill_(1.0/self.cate_num)).cuda()

        tag_onehot = Variable(torch.zeros(b_s,steps+1).cuda())
        h_t = self.init_hidden(b_s)
        for i in range(steps):
            input = inputs_p[:,i,:]
            tag = torch.sum(outputs,1, keepdim = True).long() #output: bs,step
            tag_onehot.data.zero_()
            tag_onehot.scatter_(1,tag,ones)
            i_t = torch.cat([input,tag_onehot],1) # (b_s, input+cate)
            h_t = self.gen(i_t,  h_t)
            energe_s = nn.Softmax(1)(self.cf(h_t))

            if self.training:
                action_exploit = energe_s.multinomial()  
                explorate_flag = self.switch_m.multinomial()  
                action_explorate = self.action_m.multinomial()
                action =repackage_var(explorate_flag*action_exploit +\
                    (1-explorate_flag.float().float()).long() * action_explorate) 
                # action = energe_s.multinomial()  # equivalent to line 50-54
            else:
                values,action = torch.max(energe_s,1)

            s_t =repackage_var(action.view(-1,1))
            pi = torch.gather(energe_s, 1, s_t) # (b_s,1), \pi for i,j
            pis[:,i] = pi # bs,10
            outputs[:,i] = s_t # bs,10
        return  outputs, pis

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(batch_size, self.hidden_size).zero_())

class RiboRL(nn.Module): # inheriting from nn.Module!
    
    def __init__(self,  types= 'max', n_labels = 2, **kwargs):
        super(RiboRL, self).__init__()

        seed = kwargs['seed']
        torch.manual_seed(seed)
        self.types = types
        self.kwargs = kwargs
        self.embsize = kwargs['embsize']
        self.drate = kwargs['drate']
        self.n_repeat = kwargs['n_repeat']
        self.n_hids = kwargs['n_hids']
        self.hidden_size = self.n_hids
        self.vocab_size = kwargs['n_tokens']
        self.n_filter = kwargs['n_filter']
        self.filter_list = kwargs['filter_list']
        self.length = int(kwargs['window'])
        self.L = kwargs['L']
        self.mask_position = kwargs['mask_position']

        self.lambda1 =  kwargs['lambda1']
        self.lambda2 =0# self.lambda1

        self.num_labels = n_labels
        self.boe_tag_cate_num = 2

        self.bigru_alpha = nn.GRU(self.vocab_size, self.n_hids, 1, batch_first =True, bidirectional=True)
        self.policynet = PolicyNet(2*self.n_hids, self.n_hids, self.boe_tag_cate_num, self.length)
        self.bigru_beta = nn.GRU(self.vocab_size, self.n_hids,1,batch_first =True, bidirectional=True)
        self.n_fc = self.n_hids*2

        self.drop = nn.Dropout(self.drate)
        self.fc1 = nn.Linear(self.n_fc,1) # fully connect
        print('The RiboRL has been built!')
        
    def forward(self, x, target):
        '''
        x: b_s,len
        '''

        '''encoding'''
        x = onehot(x, self.vocab_size) # bs, len, 64

        if self.training:
            self.n_repeat = self.kwargs['n_repeat']
        else:
            self.n_repeat = 1

        batch_size = x.size()[0]
        length = x.size()[1]
        inputs_r = x.repeat(self.n_repeat,1,1) # (b_s*10,l,emb_size)
        targets_r = target.view(-1,1).repeat(self.n_repeat,1)

        '''birnn_alpha'''

        self.bigru_alpha.flatten_parameters()
        self.bigru_beta.flatten_parameters()

        h0 = Variable(torch.zeros(2, batch_size*self.n_repeat, self.hidden_size)).cuda()
        h_features,hidden0 = self.bigru_alpha(inputs_r, h0) #bs*n_repeat,l,n_hids*2

        s, pis = self.policynet(h_features) # bs*n_repeat ,10

        mask_words = s.unsqueeze(2).repeat(1,1,self.vocab_size) * inputs_r #bs*n_repeat,len,64+10
        forward_features1,hidden0 = self.bigru_beta(mask_words, h0) # bs*n_repeat,len,2*n_hids

        rnn_o = nn.MaxPool1d(forward_features1.size(1))(forward_features1.permute(0,2,1))
        rnn_o = rnn_o.squeeze() #bs*n_repeat,2*n_hids

        fc_feed = self.drop(rnn_o)
        logit = self.fc1(fc_feed) # bs*n_repeat,1
        square =  (logit - targets_r)*(logit - targets_r) # bs*n_repeat,1
        '''supervised learning'''
        self.squareloss = torch.sum(square)/self.n_repeat

        '''rl'''
        R_l1 =  -self.lambda1 * torch.abs(torch.norm(s,1,1,keepdim = True)-self.L)
        reward = -square + R_l1 #  bs*n_repeat,1
        avg_reward = torch.mean(reward.view(-1,batch_size),0).repeat(self.n_repeat,1).view(-1,1)
        #bs*n_repeat,1
        real_reward = nn.ReLU()(reward - avg_reward) # bs*n_repeat
        real_reward = repackage_var(real_reward)
        rlloss = -torch.mean( torch.log(pis.clamp(1e-6,1)),1,keepdim=True)*real_reward
        self.rlloss = torch.sum(rlloss)/self.n_repeat
        eta = 0.5
        self.loss = eta*self.squareloss + (1-eta)*self.rlloss
        self.mse = torch.mean(square)
        self.reward = torch.mean(reward)
        rationales = s.long() 

        return  self.loss, self.reward, self.mse, (logit, rationales)
      
def repackage_var(vs, requires_grad = False):

    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(vs) == Variable:
        return Variable(vs.data, requires_grad = requires_grad)
    elif type(vs) == Parameter:
        return Parameter(vs.data,requires_grad = requires_grad)
    elif type(vs) == torch.Tensor:
        return Variable(vs.data,requires_grad = requires_grad)
    else:
        return tuple(repackage_var(v) for v in vs)

def onehot(data1, n_dimension):
    n_dim = data1.dim()
    batch_size = data1.size()[0]
    data = data1.view(-1,1)  
    if hasattr(data1,'data'):
        assert  (torch.max(data1)< n_dimension).data.all() # bs,1
        y_onehot = Variable(torch.FloatTensor(data.size(0),n_dimension).zero_())
        ones = Variable(torch.FloatTensor(data.size()).fill_(1))
    else:
        y_onehot = torch.FloatTensor(data.size(0),n_dimension).zero_()
        ones = torch.FloatTensor(data.size()).fill_(1)

    if data.is_cuda:
        y_onehot = y_onehot.cuda()
        ones = ones.cuda()

    y_onehot.scatter_(1,data,ones)
    if n_dim ==1:
        return y_onehot.view(batch_size,n_dimension)
    elif n_dim ==2:
        return y_onehot.view(batch_size,-1,n_dimension)







