import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,HypergraphConv, global_max_pool, global_mean_pool
from utils import reset
import tensorly as tl
tl.set_backend('pytorch')
###利用GCN从G_sim,H_sim中获取特征，利用HGCN从G-H-W中获取特征

class HgnnEncoder(torch.nn.Module):
    def __init__(self,in_channels, out_channels,args):
        super(HgnnEncoder, self).__init__()
        self.conv_HY = nn.ModuleList([HypergraphConv(in_channels, out_channels)])
        self.batch_HY = nn.ModuleList([nn.BatchNorm1d(out_channels)])
        for i in range(args.nlayer-1):
            self.conv_HY.append(HypergraphConv(out_channels, out_channels))
            self.batch_HY.append(nn.BatchNorm1d(out_channels))
        self.act = nn.Tanh()


    def forward(self, x, args):
        x_HY = []
        for i in range(args.nlayer):
            xi = self.batch_HY[i](self.act(self.conv_HY[i](x, args.edge_HGW,args.weight)))
            x = xi
            x_HY.append(xi)

        return x_HY

#将drug的信息转化为特征,mic、dis的相似性转化为相同长度的向量
class GcnEncoder(nn.Module):
    def __init__(self,dim_H, dim_W, output,args):
        super(GcnEncoder, self).__init__()

        self.drug_inf = args.durg_inf
        dim_G = self.drug_inf.x.shape[1]

        #-------G_layer  （drug）
        self.use_GMP = args.use_GMP
        self.conv_g = nn.ModuleList([GCNConv(dim_G, args.rank)])  #这个表示输入层
        self.batch_g = nn.ModuleList([nn.BatchNorm1d(args.rank)])
        for i in range(args.nlayer-1):
            self.conv_g.append(GCNConv(args.rank, args.rank))
            self.batch_g.append(nn.BatchNorm1d(args.rank))


        # -------H_layer  (mic)
        self.conv_h = nn.ModuleList([GCNConv(dim_H, args.rank)])  #这个表示输入层
        self.batch_h = nn.ModuleList([nn.BatchNorm1d(args.rank)])
        for i in range(args.nlayer-1):
            self.conv_h.append(GCNConv(args.rank, args.rank))
            self.batch_h.append(nn.BatchNorm1d(args.rank))


        # -------W_layer  (dis)
        self.conv_w = nn.ModuleList([GCNConv(dim_W, args.rank)])  #这个表示输入层
        self.batch_w = nn.ModuleList([nn.BatchNorm1d(args.rank)])
        for i in range(args.nlayer-1):
            self.conv_w.append(GCNConv(args.rank, args.rank))
            self.batch_w.append(nn.BatchNorm1d(args.rank))
        self.act = nn.Tanh()

    #对于药物，不用逐个图进行GNN，而是合并后的大图进行GNN
    def forward(self, H_feature,W_feature, args):

        G_feature, edge_G, batch_G = self.drug_inf.x, self.drug_inf.edge_index, self.drug_inf.batch
        # -----G_train   (drug)
        x_G = []
        for i in range(args.nlayer):
            x_Gi = self.batch_g[i](self.act(self.conv_g[i](G_feature, edge_G)))
            G_feature = x_Gi
            if self.use_GMP:
                x_Gii = global_max_pool(x_Gi, batch_G)
                x_G.append(x_Gii)
            else:
                x_Gii = global_mean_pool(x_Gi, batch_G)
                x_G.append(x_Gii)

        # -----H_train   (mic)
        x_H = []
        for i in range(args.nlayer):
            x_Hi = self.batch_h[i](self.act(self.conv_h[i](H_feature, args.edge_H, args.w_H)))
            H_feature = x_Hi
            x_H.append(x_Hi)


        # ----W_train    (dis)
        x_W = []
        for i in range(args.nlayer):
            x_Wi = self.batch_w[i](self.act(self.conv_w[i](W_feature, args.edge_W, args.w_W)))
            W_feature = x_Wi
            x_W.append(x_Wi)

        return x_G,x_H,x_W

class Hybridgraphattention(torch.nn.Module):
    def __init__(self, gcn_encoder, hgnn_encoder,args):
        super(Hybridgraphattention, self).__init__()
        self.gcn_encoder = gcn_encoder
        self.hgnn_encoder = hgnn_encoder
        if args.triple:
            N_fea = 2*args.nlayer
        else:
            N_fea = args.nlayer

        self.G_weight = nn.Parameter(torch.ones(N_fea))  #根据潜在特征计算相似性的时候使用
        self.H_weight = nn.Parameter(torch.ones(N_fea))
        self.W_weight = nn.Parameter(torch.ones(N_fea))
        self.reset_parameters()
        self.act = nn.Tanh()
    def reset_parameters(self):
        reset(self.gcn_encoder)
        reset(self.hgnn_encoder)

    def forward(self,args):
        Ng,Nh,Nw = args.G_num,args.H_num,args.W_num
        R = args.rank   #输入特征的维度

        ###计算这些相似性的低维特征
        torch.manual_seed(1)
        G_fea = torch.randn(Ng, R).to(args.device)
        H_fea = torch.randn(Nh, R).to(args.device)
        W_fea = torch.randn(Nw, R).to(args.device)

        G_emb,H_emb,W_emb = [],[],[]

        #True表示HGNN+GNN
        if args.triple == True:

            # step1：从G-H-W超图中获取特征
            GHW_fea = torch.cat((G_fea,H_fea,W_fea),dim=0)
            GHW_embed = self.hgnn_encoder(GHW_fea, args)   #获取两层特征
            for i in range(args.nlayer):
                G_emb1, H_emb1, W_emb1 = GHW_embed[i][:Ng], GHW_embed[i][Ng:Ng +Nh], GHW_embed[i][Ng + Nh:]
                G_emb.append(G_emb1)
                H_emb.append(H_emb1)
                W_emb.append(W_emb1)

            # step2：从G,H的相似性图中获取特征
            x_G,x_H,x_W = self.gcn_encoder(H_fea,W_fea, args)
            for i in range(args.nlayer):
                G_emb.append(x_G[i])
                H_emb.append(x_H[i])
                W_emb.append(x_W[i])


            G_weight = torch.exp(self.G_weight)/torch.exp(self.G_weight).sum()
            H_weight = torch.exp(self.H_weight)/torch.exp(self.H_weight).sum()
            W_weight = torch.exp(self.W_weight)/torch.exp(self.W_weight).sum()
            G_emb_all = torch.stack([G_emb[i] * G_weight[i] for i in range(len(G_emb))]).sum(dim=0)
            H_emb_all = torch.stack([H_emb[i] * H_weight[i] for i in range(len(H_emb))]).sum(dim=0)
            W_emb_all = torch.stack([W_emb[i] * W_weight[i] for i in range(len(W_emb))]).sum(dim=0)
            graph_embed = torch.cat((G_emb_all,H_emb_all,W_emb_all),dim=0)
        else:  #否则只使用GNN
            x_G,x_H,x_W = self.gcn_encoder(H_fea,W_fea, args)
            for i in range(args.nlayer):
                G_emb.append(x_G[i])
                H_emb.append(x_H[i])
                W_emb.append(x_W[i])

            G_weight = torch.exp(self.G_weight)/torch.exp(self.G_weight).sum()
            H_weight = torch.exp(self.H_weight)/torch.exp(self.H_weight).sum()
            W_weight = torch.exp(self.W_weight)/torch.exp(self.W_weight).sum()
            G_emb = torch.stack([G_emb[i] * G_weight[i] for i in range(len(G_emb))]).sum(dim=0)
            H_emb = torch.stack([H_emb[i] * H_weight[i] for i in range(len(H_emb))]).sum(dim=0)
            W_emb = torch.stack([W_emb[i] * W_weight[i] for i in range(len(W_emb))]).sum(dim=0)
            graph_embed = torch.cat((G_emb, H_emb, W_emb), dim=0)


        return graph_embed