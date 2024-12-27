import torch
import numpy as np
from torch_scatter import scatter

import argparse


def parse():
    p = argparse.ArgumentParser("BayesHGNN: 贝叶斯超图神经网络", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', type=str, default='DATA1_zhang', help='data name ')
    p.add_argument('--dataset', type=str, default='./DATA', help='dataset name')
    p.add_argument('--model-name', type=str, default='VBMGD', help='贝叶斯混合图神经网络')
    p.add_argument('--activation', type=str, default='tanh', help='activation layer between MGConvs')
    p.add_argument('--nlayer', type=int, default=2, help='number of hidden layers')
    # p.add_argument('--F1', type=int, default=50, help='特征输入层的维度')
    # p.add_argument('--F2', type=int, default=20, help='隐藏层的维度')
    p.add_argument('--rank', type=int, default=10, help='子空间的维度')
    # p.add_argument('--N-num', type=int, default=10, help='邻居个数')
    p.add_argument('--lr', type=float, default=0.001, help='learning rate')
    p.add_argument('--L2',type=float,default=1e-4,help = 'weight_decay')
    p.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    return p.parse_args()

class Metrics():
    def __init__(self, step, test_data, predict_1, batch_size, top):
        self.pair = []
        self.step = step
        self.test_data = test_data
        self.predict_1 = predict_1
        self.top = top
        self.dcgsum = 0
        self.idcgsum = 0
        self.hit = 0
        self.ndcg = 0
        self.batch_size = batch_size
        self.val_top = []

    def hits_ndcg(self):
        #这表示从self.test_data[i, ：]中提取self.step * self.batch_size, (self.step + 1) * self.batch_size
        #从self.predict_1中也提取对应位置的
        #它的作用是从self.batch_size个样本中，判断前self.top个样本是否有命中，并且计算ndcg
        for i in range(self.step * self.batch_size, (self.step + 1) * self.batch_size):
            g = []
            g.extend([self.test_data[i, 3], self.predict_1[i].item()])
            self.pair.append(g)
        np.random.seed(1)
        np.random.shuffle(self.pair)
        pre_val = sorted(self.pair, key=lambda item: item[1], reverse=True) #这个表示的意思应该是按照得分的降序进行排序
        self.val_top = pre_val[0: self.top]

        #self.hit取值为1或0，取值为1表示前self.top中有命中的，若取值为0表示前self.top中没有命中的
        #self.dcgsum=1，表示第一个命中。若命中的排名靠后，self.dcgsum越小。
        for i in range(len(self.val_top)):
            if self.val_top[i][0] == 1: #前i个里面是关联
                self.hit = self.hit + 1  #命中个数加1
                #self.top 里面有多个，若排名第一的被命中，self.dcgsum = 1/log2(0 + 2)=1
                #若排名第一的没有被命中，排名第二的被命中，则self.dcgsum = 1/log2(1 + 2)
                #若排名第二的没有被命中，排名第三的被命中，则self.dcgsum = 1/log2(2 + 2)
                self.dcgsum = (2 ** self.val_top[i][0] - 1) / np.log2(i + 2) #(2 ** self.val_top[i][0] - 1) 这应该都是1   等价于1/np.log2(i+2)
                break
        ideal_list = sorted(self.val_top, key=lambda item: item[0], reverse=True)  #按照关联进行排序，关联的排在前面
        for i in range(len(ideal_list)):
            if ideal_list[i][0] == 1:
                self.idcgsum = (2 ** ideal_list[i][0] - 1) / np.log2(i + 2)
                break
        if self.idcgsum == 0:
            self.ndcg = 0
        else:
            self.ndcg = self.dcgsum / self.idcgsum  #感觉这个self.idcgsum没用，self.idcgsum只能取0或1
        if self.ndcg != self.dcgsum:
            print('程序出错！'*10)
        return self.hit, self.ndcg


def hit_ndcg_value(pred_val, val_data, top):
    loader_val = torch.utils.data.DataLoader(dataset=pred_val, batch_size=30, shuffle=False)
    hits = 0
    ndcg_val = 0
    for step, batch_val in enumerate(loader_val):
        metrix = Metrics(step, val_data, pred_val, batch_size=30, top=top)
        hit, ndcg = metrix.hits_ndcg()
        hits = hits + hit
        ndcg_val = ndcg_val + ndcg
    hits = hits / int((len(val_data)) / 30) #int((len(val_data)) / 30) 这个表示组数
    ndcg = ndcg_val / int((len(val_data)) / 30)
    return hits, ndcg






def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def Loss_fun_opt(FGHW, GHW,lambdas,args):
    lambdas = lambdas.diag().to(args.device)
    Ng, Nh, Nw = args.G_num, args.H_num, args.W_num
    G_emb, H_emb, W_emb = GHW[:Ng], GHW[Ng:Ng + Nh], GHW[Ng + Nh:]
    FG_emb, FH_emb, FW_emb = FGHW[:Ng], FGHW[Ng:Ng + Nh], FGHW[Ng + Nh:]
    loss_g = torch.trace(lambdas @ (G_emb-FG_emb).t() @ (G_emb-FG_emb))
    loss_h = torch.trace(lambdas @ (H_emb - FH_emb).t() @ (H_emb - FH_emb))
    loss_w = torch.trace(lambdas @ (W_emb - FW_emb).t() @ (W_emb - FW_emb))
    return (loss_h+loss_g+loss_w)/(Ng+Nh+Nw)



def Const_hyper(args,sim_H,sim_W,train_tensor):
    #Step1:根据相似性构建图
    #sg
    for i in range(len(sim_H)):
        sim_H[i,i]=1
    sim_H = (sim_H+sim_H.t())/2
    ind_H = torch.nonzero(sim_H)
    edge_H = ind_H.t().to(args.device)
    w_H = sim_H[ind_H[:,0],ind_H[:,1]]
    args.edge_H = edge_H.to(args.device)
    args.w_H = w_H.to(args.device)

    #sh
    for i in range(len(sim_W)):
        sim_W[i,i]=1
    sim_W = (sim_W + sim_W.t()) / 2
    ind_W = torch.nonzero(sim_W)
    edge_W = ind_W.t().to(args.device)
    w_W = sim_W[ind_W[:,0],ind_W[:,1]]
    args.edge_W = edge_W.to(args.device)
    args.w_W = w_W.to(args.device)

    #sghw
    Ng,Nh,Nw = train_tensor.shape
    indice_T = torch.nonzero(train_tensor)
    indice_T[:,1] += Ng
    indice_T[:, 2] += Ng+Nh
    NT = indice_T.shape[0]
    edges_T = torch.arange(NT).unsqueeze(1).expand(NT, 3)+Ng+Nh
    hyper_3 = torch.cat((indice_T.reshape(1, NT * 3), edges_T.reshape(1, NT * 3)), dim=0)
    wghw = torch.cat([torch.ones(1,3) for idx in indice_T],dim=1).squeeze()


    synergy_graph = hyper_3
    #model2中的两个参数 V和E
    args.edge_HGW = synergy_graph.to(args.device)
    args.weight = wghw.to(args.device)


    return args


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)  # 超边矩阵
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)  # 超边权重矩阵
    # the degree of the node
    DV = np.sum(H * W, axis=1)  # 节点度; (12311,)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # 超边的度; (24622,)

    invDE = np.mat(np.diag(np.power(DE, -1)))  # DE^-1; 建立对角阵
    invDE[np.isinf(invDE)] = 0
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2[np.isinf(DV2)] = 0# DV^-1/2
    W = np.mat(np.diag(W))  # 超边权重矩阵
    H = np.mat(H)  # 超边矩阵
    HT = H.T

    if variable_weight:  # 超边权重是否可变
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G