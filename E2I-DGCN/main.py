import torch
import torch.optim as optim
import scipy
import torch.nn.functional as F
import numpy as np

from model import E2IDGCN
from utility.helper import *
from utility.batch_test import *
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')
from time import time

torch.cuda.empty_cache()

if __name__ == '__main__':


    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj_G, norm_adj_G1, norm_adj_G2 = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    # args.mess_dropout = eval(args.mess_dropout)


    model = E2IDGCN(data_generator.n_users,
                  data_generator.n_items,
                  norm_adj_G,
                  norm_adj_G1,
                  norm_adj_G2,
                  args).to(args.device)

    t0 = time()
    """    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users_G, pos_items_G, neg_items_G, users_G1, pos_items_G1, neg_items_G1, users_G2, pos_items_G2, neg_items_G2 = data_generator.sample()
            u_g_embeddings_G, pos_i_g_embeddings_G, neg_i_g_embeddings_G, u_g_embeddings_G1, pos_i_g_embeddings_G1, neg_i_g_embeddings_G1, u_g_embeddings_G2, pos_i_g_embeddings_G2, neg_i_g_embeddings_G2 = model(users_G,
                                                                           pos_items_G,
                                                                           neg_items_G, users_G1, pos_items_G1, neg_items_G1, users_G2, pos_items_G2, neg_items_G2,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings_G,
                                                                              pos_i_g_embeddings_G,
                                                                              neg_i_g_embeddings_G,
                                                                              u_g_embeddings_G1,
                                                                              pos_i_g_embeddings_G1,
                                                                              neg_i_g_embeddings_G1,
                                                                              u_g_embeddings_G2,
                                                                              pos_i_g_embeddings_G2,
                                                                              neg_i_g_embeddings_G2,
                                                                              )
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                       'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3],
                        ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][3], ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][3],
                        ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][3], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        if should_stop == True:
            break


    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 3])
    idx = list(recs[:, 3]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)