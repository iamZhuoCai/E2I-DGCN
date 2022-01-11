import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.helper import *
from utility.batch_test import *
# from main import pre_para
from torch.autograd import Variable
import numpy as np
import scipy.sparse as sp
import random as rd


class E2IDGCN(nn.Module):
    def __init__(self, n_user, n_item, norm_adj_G, norm_adj_G1, norm_adj_G2, args):
        super(E2IDGCN, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]

        self.norm_adj_G = norm_adj_G
        self.norm_adj_G1 = norm_adj_G1
        self.norm_adj_G2 = norm_adj_G2

        self.layers = eval(args.layer_size)
        self.alpha = eval(args.alpha)
        self.beta = args.beta
        self.decay = eval(args.regs)[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj of user-item bipartite graph and intent-aware subgraphs.
        """
        self.sparse_norm_adj_G = self._convert_sp_mat_to_sp_tensor(self.norm_adj_G).to(self.device)
        self.sparse_norm_adj_G1 = self._convert_sp_mat_to_sp_tensor(self.norm_adj_G1).to(self.device)
        self.sparse_norm_adj_G2 = self._convert_sp_mat_to_sp_tensor(self.norm_adj_G2).to(self.device)


    def init_weight(self):

        initializer = nn.init.xavier_uniform_


        embedding_dict = nn.ParameterDict({
            # user and item embeddings
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size))),


            # edge embeddings
            'edge_emb_G': nn.Parameter(initializer(torch.empty(1,
                                                                  self.emb_size))),
            'edge_emb_G1': nn.Parameter(initializer(torch.empty(1,
                                                                  self.emb_size))),
            'edge_emb_G2': nn.Parameter(initializer(torch.empty(1,
                                                                   self.emb_size)))
        })


        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            # feature transformation matirxes for edge embeddings
            weight_dict.update({'W_edge_G_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_edge_G_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_edge_G1_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_edge_G1_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_edge_G2_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_edge_G2_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users_G, pos_items_G, neg_items_G, u_g_embeddings_G1, pos_i_g_embeddings_G1, neg_i_g_embeddings_G1, u_g_embeddings_G2, pos_i_g_embeddings_G2, neg_i_g_embeddings_G2):
        # preference scores in G, G1, G2
        pos_scores_G = torch.sum((torch.mul(users_G, pos_items_G)), axis=1)
        neg_scores_G = torch.sum((torch.mul(users_G, neg_items_G)), axis=1)

        pos_scores_G1 = torch.sum((torch.mul(u_g_embeddings_G1, pos_i_g_embeddings_G1)), axis=1)
        neg_scores_G1 = torch.sum((torch.mul(u_g_embeddings_G1, neg_i_g_embeddings_G1)), axis=1)

        pos_scores_G2 = torch.sum((torch.mul(u_g_embeddings_G2, pos_i_g_embeddings_G2)), axis=1)
        neg_scores_G2 = torch.sum((torch.mul(u_g_embeddings_G2, neg_i_g_embeddings_G2)), axis=1)


        # Loss: BPR + similarity + regularizer
        score_G = nn.LogSigmoid()(pos_scores_G - neg_scores_G) + self.beta * torch.cosine_similarity(users_G, pos_items_G, 1) - self.beta * torch.cosine_similarity(users_G, neg_items_G, 1)

        score_G1 = nn.LogSigmoid()(pos_scores_G1 - neg_scores_G1) + self.beta * torch.cosine_similarity(u_g_embeddings_G1, pos_i_g_embeddings_G1, 1) - self.beta * torch.cosine_similarity(u_g_embeddings_G1, neg_i_g_embeddings_G1, 1)

        score_G2 = nn.LogSigmoid()(pos_scores_G2 - neg_scores_G2) + self.beta * torch.cosine_similarity(u_g_embeddings_G2, pos_i_g_embeddings_G2, 1) - self.beta * torch.cosine_similarity(u_g_embeddings_G2, neg_i_g_embeddings_G2, 1)

        mf_loss = -1 * torch.mean(score_G + self.alpha[0] * score_G1 + self.alpha[1] * score_G2)

        # culculate regularizer
        regularizer = (torch.norm(users_G) ** 2
                       + torch.norm(pos_items_G) ** 2
                       + torch.norm(neg_items_G) ** 2
                       + torch.norm(u_g_embeddings_G1) ** 2
                       + torch.norm(pos_i_g_embeddings_G1) ** 2
                       + torch.norm(neg_i_g_embeddings_G1) ** 2
                       + torch.norm(u_g_embeddings_G2) ** 2
                       + torch.norm(pos_i_g_embeddings_G2) ** 2
                       + torch.norm(neg_i_g_embeddings_G2) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings_G, pos_i_g_embeddings_G, u_g_embeddings_G1, pos_i_g_embeddings_G1, u_g_embeddings_G2, pos_i_g_embeddings_G2):
        return torch.matmul(u_g_embeddings_G, pos_i_g_embeddings_G.t()) + self.alpha[0] * (torch.matmul(u_g_embeddings_G1, pos_i_g_embeddings_G1.t()) + self.alpha[1] * torch.matmul(u_g_embeddings_G2, pos_i_g_embeddings_G2.t()))



    def forward(self, users_G, pos_items_G, neg_items_G, users_G1, pos_items_G1, neg_items_G1, users_G2, pos_items_G2, neg_items_G2, drop_flag=True):
        A_hat_G = self.sparse_dropout(self.sparse_norm_adj_G,
                                    self.node_dropout,
                                    self.sparse_norm_adj_G._nnz()) if drop_flag else self.sparse_norm_adj_G

        A_hat_G1 = self.sparse_dropout(self.sparse_norm_adj_G1,
                                    self.node_dropout,
                                    self.sparse_norm_adj_G1._nnz()) if drop_flag else self.sparse_norm_adj_G1

        A_hat_G2 = self.sparse_dropout(self.sparse_norm_adj_G2,
                                    self.node_dropout,
                                    self.sparse_norm_adj_G2._nnz()) if drop_flag else self.sparse_norm_adj_G2


        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                  self.embedding_dict['item_emb']], 0)

        ego_embeddings_G1 = ego_embeddings.clone()
        ego_embeddings_G2 = ego_embeddings.clone()


        all_embeddings = ego_embeddings.clone()
        all_embeddings_G2 = ego_embeddings_G2.clone()
        all_embeddings_G1 = ego_embeddings_G1.clone()


        edge_embeddings_G = self.embedding_dict['edge_emb_G']
        edge_embeddings_G1 = self.embedding_dict['edge_emb_G1']
        edge_embeddings_G2 = self.embedding_dict['edge_emb_G2']




        for k in range(len(self.layers)):
            side_embeddings_G = torch.mul(ego_embeddings, edge_embeddings_G)
            side_embeddings_G = torch.sparse.mm(A_hat_G, side_embeddings_G)

            side_embeddings_G2 = torch.mul(ego_embeddings_G2, edge_embeddings_G2)
            side_embeddings_G2 = torch.sparse.mm(A_hat_G2, side_embeddings_G2)

            side_embeddings_G1 = torch.mul(ego_embeddings_G1, edge_embeddings_G1)
            side_embeddings_G1 = torch.sparse.mm(A_hat_G1, side_embeddings_G1)



            ego_embeddings_G = side_embeddings_G
            ego_embeddings_G1 = side_embeddings_G1
            ego_embeddings_G2 = side_embeddings_G2


            all_embeddings += ego_embeddings_G
            all_embeddings_G1 += ego_embeddings_G1
            all_embeddings_G2 += ego_embeddings_G2



            # update of edeg embeddings
            edge_embeddings_G = torch.matmul(edge_embeddings_G, self.weight_dict['W_edge_G_%d' % k]) \
                                              + self.weight_dict['b_edge_G_%d' % k]  # G

            edge_embeddings_G1 = torch.matmul(edge_embeddings_G1, self.weight_dict['W_edge_G1_%d' % k]) \
                                              + self.weight_dict['b_edge_G1_%d' % k]  # G1

            edge_embeddings_G2 = torch.matmul(edge_embeddings_G2, self.weight_dict['W_edge_G2_%d' % k]) \
                                              + self.weight_dict['b_edge_G2_%d' % k]  #G2



        u_g_embeddings_G = all_embeddings[:self.n_user, :]
        i_g_embeddings_G = all_embeddings[self.n_user:, :]

        u_g_embeddings_G1 = all_embeddings_G1[:self.n_user, :]
        i_g_embeddings_G1 = all_embeddings_G1[self.n_user:, :]

        u_g_embeddings_G2 = all_embeddings_G2[:self.n_user, :]
        i_g_embeddings_G2 = all_embeddings_G2[self.n_user:, :]



        """
        *********************************************************
        look up.
        """
        u_g_embeddings_G = u_g_embeddings_G[users_G, :]
        pos_i_g_embeddings_G = i_g_embeddings_G[pos_items_G, :]
        neg_i_g_embeddings_G = i_g_embeddings_G[neg_items_G, :]

        u_g_embeddings_G1 = u_g_embeddings_G1[users_G1, :]
        pos_i_g_embeddings_G1 = i_g_embeddings_G1[pos_items_G1, :]
        neg_i_g_embeddings_G1 = i_g_embeddings_G1[neg_items_G1, :]

        u_g_embeddings_G2 = u_g_embeddings_G2[users_G2, :]
        pos_i_g_embeddings_G2 = i_g_embeddings_G2[pos_items_G2, :]
        neg_i_g_embeddings_G2 = i_g_embeddings_G2[neg_items_G2, :]




        return u_g_embeddings_G, pos_i_g_embeddings_G, neg_i_g_embeddings_G, u_g_embeddings_G1, pos_i_g_embeddings_G1, neg_i_g_embeddings_G1, u_g_embeddings_G2, pos_i_g_embeddings_G2, neg_i_g_embeddings_G2
