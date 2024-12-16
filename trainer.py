import argparse
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR

from data_loader import Data_loader
from evaluator import evaluator as ev
from ponegnn import PoneGNN
from logger import Logger
from util import *

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='ML-1M',
                        help="Dataset"
                        )
    parser.add_argument('--version',
                        type=int,
                        default=1,
                        help="Dataset version"
                        )
    parser.add_argument('--batch_size',
                        type=int,
                        default=2048,
                        help="Batch size"
                        )

    parser.add_argument('--dim',
                        type=int,
                        default=64,
                        help="Dimension"
                        )
    parser.add_argument('--lr',
                        type=float,
                        default=5e-3,
                        help="Learning rate"
                        )
    parser.add_argument('--offset',
                        type=float,
                        default=3.5,
                        help="Criterion of likes/dislikes"
                        )
    parser.add_argument('--K',
                        type=int,
                        default=40,
                        help="The number of negative samples"
                        )
    parser.add_argument('--num_layer',
                        type=int,
                        default=4,
                        help="The number of layers of a GNN model for the graph with positive edges"
                        )
    parser.add_argument('--MLP_layers',
                        type=int,
                        default=2,
                        help="The number of layers of MLP for the graph with negative edges"
                        )
    parser.add_argument('--epoch',
                        type=int,
                        default=201,
                        help="The number of epochs"
                        )
    parser.add_argument('--reg',
                        type=float,
                        default=5e-5,  # ,0.05
                        help="Regularization coefficient"
                        )
    parser.add_argument('--aggregate',
                        type=str,
                        default='pandgnn',
                        help="aggregate method"
                        )
    parser.add_argument('--freq',
                        type=int,
                        default=20,
                        help="valid frequency"
                        )
    return parser.parse_args()


def mean(lst):
    s = sum(lst)
    n = len(lst)
    return s / n


def sd(lst):
    mean = sum(lst) / len(lst)
    bias_mean = [(x - mean) ** 2 for x in lst]
    s2 = sum(bias_mean) / len(bias_mean)
    return math.sqrt(s2)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


class Trainer(object):
    def __init__(self, args):
        self.args = args
        task_name = "%s_%s%s" % (datetime.now().strftime('%m%d%H%M'), args.dataset, args.version)
        self.logger = Logger(task_name, False)
        self.logger.logging(str(args))

    def train(self):
        os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
        args = self.args
        all_r = []
        all_p = []
        all_n = []
        all_h = []
        for index in range(1):
            self.logger.logging("Current index: %s" % index)
            data_class = Data_loader(args.dataset, args.version)
            self.logger.logging('data loading...')
            st = time.time()
            train, test = data_class.data_load()

            train['userId'].fillna(0, inplace=True)
            train['movieId'].fillna(0, inplace=True)
            train = train.astype({'userId': 'int64', 'movieId': 'int64'})
            data_class.train = train
            data_class.test = test
            self.logger.logging('Loading complete! time :: %s' % (time.time() - st))

            self.logger.logging('Generate negative candidates...')
            st = time.time()
            if args.dataset == 'ML-1M':
                neg_dist = deg_dist(train, data_class.num_v)
            else:
                neg_dist = deg_dist_2(train, data_class.num_v)
            self.logger.logging('complete ! time : %s' % (time.time() - st))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = LightSignedGCN(data_class.num_u, data_class.num_v, num_layer=args.num_layer, dim=args.dim,
                                   reg=args.reg)
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = MultiStepLR(optimizer, milestones=[20, 200], gamma=0.2)

            self.logger.logging("Training on {}...".format(device))
            model.train()

            training_dataset = bipartite_dataset(train, neg_dist, args.offset, data_class.num_u, data_class.num_v,
                                                 args.K)

            res_r, res_p, res_n, res_h = 0, 0, 0, 0

            # Positive graph
            # args.offset = 0  # for traditional CF methods
            edge_user = torch.tensor(train[train['rating'] > args.offset]['userId'].values - 1)  # Index from 0
            edge_item = torch.tensor(train[train['rating'] > args.offset]['movieId'].values - 1) + data_class.num_u
            edge_p = torch.stack((torch.cat((edge_user, edge_item), 0), torch.cat((edge_item, edge_user), 0)), 0)

            data_p = Data(edge_index=edge_p)
            data_p.to(device)

            # Negative graph
            offset_n = 3.5
            edge_user_n = torch.tensor(train[train['rating'] < offset_n]['userId'].values - 1)
            edge_item_n = torch.tensor(train[train['rating'] < offset_n]['movieId'].values - 1) + data_class.num_u
            edge_n = torch.stack((torch.cat((edge_user_n, edge_item_n), 0), torch.cat((edge_item_n, edge_user_n), 0)), 0)
            data_n = Data(edge_index=edge_n)
            data_n.to(device)

            self.logger.logging(data_n.edge_index.shape[1] / data_p.edge_index.shape[1])

            for epo in range(1, args.epoch + 1):
                if epo % args.freq - 1 == 0:
                    training_dataset.negs_gen_EP(20)

                total_loss = 0
                training_dataset.edge_4 = training_dataset.edge_4_tot[:, :, epo % args.freq - 1]

                ds = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
                q = 0
                pbar = tqdm(desc='Version : {} Epoch {}/{}'.format(args.version, epo, args.epoch), total=len(ds),
                            position=0)

                for u, v, w, negs in ds:
                    u, v, w, negs = u.to(device), v.to(device), w.to(device), negs.to(device)
                    q += len(u)
                    optimizer.zero_grad()
                    loss = model.loss(u, v, w, negs, data_p, data_n, epo)  # original
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(ds)

                    pbar.update(1)
                    pbar.set_postfix({'loss': loss.item()})

                pbar.close()
                scheduler.step()

                if epo % args.freq == 1:

                    model.eval()
                    if args.aggregate == 'siren':
                        emb = model.aggregate()
                        emb_u, emb_v = torch.split(emb, [data_class.num_u, data_class.num_v])
                        emb_u = emb_u.cpu().detach()
                        emb_v = emb_v.cpu().detach()
                        r_hat = emb_u.mm(emb_v.t())
                        reco = gen_top_k(data_class, r_hat)
                    else:
                        emb_u, emb_n_u, emb_v, emb_n_v = model.get_ui_embeddings(data_p, data_n)
                        r_hat = emb_u.mm(emb_v.t()).cpu().detach()
                        r_hat_n = emb_n_u.mm(emb_n_v.t()).cpu().detach()
                        # reco = gen_top_k(data_class, r_hat)  # no filter
                        reco = gen_top_k_new3(data_class, r_hat, r_hat_n)

                    eval_ = ev(data_class, reco, args)
                    eval_.precision_and_recall()
                    eval_.normalized_DCG()
                    self.logger.logging("***************************************************************************************")
                    self.logger.logging(" /* Recommendation Accuracy */")
                    self.logger.logging('N :: %s' % eval_.N)
                    self.logger.logging('Precision at :: %s %s' % (eval_.N, eval_.p['total'][eval_.N - 1]))
                    self.logger.logging('Recall at :: %s %s' % (eval_.N, eval_.r['total'][eval_.N - 1]))
                    self.logger.logging('nDCG at :: %s %s' % (eval_.N, eval_.nDCG['total'][eval_.N - 1]))
                    self.logger.logging('Hit at :: %s %s' % (eval_.N, eval_.h['total'][eval_.N - 1]))
                    self.logger.logging('TP at :: %s %s' % (eval_.N, eval_.tp['total'][eval_.N - 1]))
                    self.logger.logging('FP at :: %s %s' % (eval_.N, eval_.fp['total'][eval_.N - 1]))
                    self.logger.logging('FN at :: %s %s' % (eval_.N, eval_.fn['total'][eval_.N - 1]))
                    self.logger.logging("***************************************************************************************")
                    if eval_.r['total'][eval_.N - 1][2] > res_r:
                        res_r = eval_.r['total'][eval_.N - 1][2]
                        res_p = eval_.p['total'][eval_.N - 1][2]
                        res_n = eval_.nDCG['total'][eval_.N - 1][2]
                        res_h = eval_.h['total'][eval_.N - 1][2]
                    model.train()

            if epo == args.epoch:
                self.logger.logging("Final results (R,P,N,H) are: [%s %s %s %s]" % (res_r, res_p, res_n, res_h))

            all_r.append(res_r)
            all_p.append(res_p)
            all_n.append(res_n)
            all_h.append(res_h)
            torch.save(model, "./checkpoints/%s_model.pth" % args.dataset)

        self.logger.logging("Finish!")
        # self.logger.logging("mean: ", mean(all_r), mean(all_p), mean(all_n), mean(all_h))
        # self.logger.logging("sd: ", sd(all_r), sd(all_p), sd(all_n), sd(all_h))


if __name__ == "__main__":
    arg = arg_parse()
    setup_seed(3407)  # 3407
    trainner = Trainer(arg)
    trainner.train()
