import time

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class bipartite_dataset(Dataset):
    def __init__(self, train, neg_dist, offset, num_u, num_v, K):
        self.edge_1 = torch.tensor(train['userId'].values - 1)
        self.edge_2 = torch.tensor(train['movieId'].values - 1) + num_u

        # self.edge_1 = torch.tensor(train[train['rating']>offset]['userId'].values-1)
        # self.edge_2 = torch.tensor(train[train['rating']>offset]['movieId'].values-1) +num_u

        self.edge_3 = torch.tensor(train['rating'].values) - offset
        self.neg_dist = neg_dist
        self.K = K
        self.num_u = num_u
        self.num_v = num_v
        self.tot = np.arange(num_v)
        self.train = train
        self.offset = offset

    def negs_gen_(self):
        print('negative sampling...');
        st = time.time()
        self.edge_4 = torch.empty((len(self.edge_1), self.K), dtype=torch.long)
        prog = tqdm(desc='negative sampling for each epoch...', total=len(set(self.train['userId'].values)), position=0)
        for j in set(self.train['userId'].values):
            pos = self.train[self.train['userId'] == j]['movieId'].values - 1
            neg = np.setdiff1d(self.tot, pos)
            temp = (torch.tensor(np.random.choice(neg, len(pos) * self.K, replace=True,
                                                  p=self.neg_dist[neg] / self.neg_dist[neg].sum())) + self.num_u).long()
            self.edge_4[self.edge_1 == j - 1] = temp.view(int(len(temp) / self.K), self.K)
            prog.update(1)
        prog.close()
        self.edge_4 = torch.tensor(self.edge_4).long()
        print('comlete ! %s' % (time.time() - st))

    def negs_gen_EP(self, epoch):  # pos > non_feedback; neg > non_feedback
        print('negative sampling for next epochs...');
        st = time.time()
        self.edge_4_tot = torch.empty((len(self.edge_1), self.K, epoch), dtype=torch.long)
        prog = tqdm(desc='negative sampling for next epochs...', total=len(set(self.train['userId'].values)),
                    position=0)
        for j in set(self.train['userId'].values):
            pos = self.train[self.train['userId'] == j]['movieId'].values - 1
            neg = np.setdiff1d(self.tot, pos)
            temp = (torch.tensor(np.random.choice(neg, len(pos) * self.K * epoch, replace=True,
                                                  p=self.neg_dist[neg] / self.neg_dist[neg].sum())) + self.num_u).long()
            self.edge_4_tot[self.edge_1 == j - 1] = temp.view(int(len(temp) / self.K / epoch), self.K, epoch)
            prog.update(1)
        prog.close()
        self.edge_4_tot = torch.tensor(self.edge_4_tot).long()
        print('comlete ! %s' % (time.time() - st))

    def negs_gen_EP_lightgcn(self, epoch):  # pos > non_feedback
        print('negative sampling for next epochs...');
        st = time.time()
        self.edge_4_tot = torch.empty((len(self.edge_1), self.K, epoch), dtype=torch.long)
        prog = tqdm(desc='negative sampling for next epochs...', total=len(set(self.train['userId'].values)),
                    position=0)
        for j in set(self.train['userId'].values):
            pos = self.train[self.train['userId'] == j][self.train['rating'] > self.offset]['movieId'].values - 1
            neg = np.setdiff1d(self.tot, pos)
            temp = (torch.tensor(np.random.choice(neg, len(pos) * self.K * epoch, replace=True,
                                                  p=self.neg_dist[neg] / self.neg_dist[neg].sum())) + self.num_u).long()
            self.edge_4_tot[self.edge_1 == j - 1] = temp.view(int(len(temp) / self.K / epoch), self.K, epoch)
            prog.update(1)
        prog.close()
        self.edge_4_tot = torch.tensor(self.edge_4_tot).long()
        print('comlete ! %s' % (time.time() - st))

    def negs_gen_EP_double_bpr(self, epoch):  # pos > neg; neg > non_feedback
        print('negative sampling for next epochs...');
        st = time.time()
        self.edge_4_tot = torch.empty((len(self.edge_1), self.K, epoch), dtype=torch.long)
        self.edge_5_tot = torch.empty((len(self.edge_1), self.K, epoch), dtype=torch.long)
        prog = tqdm(desc='negative sampling for next epochs...', total=len(set(self.train['userId'].values)),
                    position=0)
        for j in set(self.train['userId'].values):
            pos = self.train[self.train['userId'] == j][self.train['rating'] > self.offset]['movieId'].values - 1
            neg = self.train[self.train['userId'] == j][self.train['rating'] < self.offset]['movieId'].values - 1
            if neg.size == 0:
                continue
            non_fb = np.setdiff1d(self.tot, pos)
            temp = (torch.tensor(np.random.choice(neg, len(pos) * self.K * epoch, replace=True,
                                                  p=self.neg_dist[neg] / self.neg_dist[neg].sum())) + self.num_u).long()
            temp2 = (torch.tensor(np.random.choice(non_fb, len(pos) * self.K * epoch, replace=True,
                                                   p=self.neg_dist[non_fb] / self.neg_dist[
                                                       non_fb].sum())) + self.num_u).long()
            self.edge_4_tot[self.edge_1 == j - 1] = temp.view(int(len(temp) / self.K / epoch), self.K, epoch)
            self.edge_5_tot[self.edge_1 == j - 1] = temp2.view(int(len(temp2) / self.K / epoch), self.K, epoch)
            prog.update(1)
        prog.close()
        self.edge_4_tot = torch.tensor(self.edge_4_tot).long()
        self.edge_5_tot = torch.tensor(self.edge_5_tot).long()
        print('comlete ! %s' % (time.time() - st))

    def __len__(self):
        return len(self.edge_1)

    def __getitem__(self, idx):
        u = self.edge_1[idx]
        v = self.edge_2[idx]
        w = self.edge_3[idx]
        negs = self.edge_4[idx]
        return u, v, w, negs


def deg_dist(train, num_v):
    uni, cou = np.unique(train['movieId'].values - 1, return_counts=True)
    cou = cou ** (0.75)
    deg = np.zeros(num_v)
    deg[uni] = cou
    return torch.tensor(deg)


def deg_dist_2(train, num_v):
    uni, cou = np.unique(train['movieId'].values - 1, return_counts=True)
    cou = cou ** (0.75)
    deg = np.zeros(num_v)
    deg[uni] = cou
    return torch.tensor(deg)


def gen_top_k(data_class, r_hat, K=300):
    all_items = set(np.arange(1, data_class.num_v + 1))
    tot_items = set(data_class.train['movieId']).union(set(data_class.test['movieId']))
    no_items = all_items - tot_items
    tot_items = torch.tensor(list(tot_items)) - 1
    no_items = (torch.tensor(list(no_items)) - 1).long()
    r_hat[:, no_items] = -np.inf
    for u, i in data_class.train.values[:, :-1] - 1:
        r_hat[u, i] = -np.inf

    _, reco = torch.topk(r_hat, K)
    reco = reco.numpy()

    return reco


def gen_top_k_new(data_class, r_hat_p, r_hat_n, K=300):
    all_items = set(np.arange(1, data_class.num_v + 1))
    tot_items = set(data_class.train['movieId']).union(set(data_class.test['movieId']))
    no_items = all_items - tot_items
    tot_items = torch.tensor(list(tot_items)) - 1
    no_items = (torch.tensor(list(no_items)) - 1).long()
    r_hat_p[:, no_items] = -np.inf
    for u, i in data_class.train.values[:, :-1] - 1:
        r_hat_p[u, i] = -np.inf

    _, reco_n = torch.topk(r_hat_n, K)

    for u in range(reco_n.shape[0]):
        for i in range(3):
            idx = reco_n[u, i]
            r_hat_p[u, idx] = -np.inf

    _, reco = torch.topk(r_hat_p, K)
    reco = reco.numpy()

    return reco


def gen_top_k_new2(data_class, r_hat_p, r_hat_n, K=300):
    all_items = set(np.arange(1, data_class.num_v + 1))
    tot_items = set(data_class.train['movieId']).union(set(data_class.test['movieId']))
    no_items = all_items - tot_items
    tot_items = torch.tensor(list(tot_items)) - 1
    no_items = (torch.tensor(list(no_items)) - 1).long()
    r_hat_p[:, no_items] = -np.inf
    for u, i in data_class.train.values[:, :-1] - 1:
        r_hat_p[u, i] = -np.inf

    _, reco = torch.topk(r_hat_p, 20)
    reco_score_p = torch.zeros([reco.shape[0], 20], dtype=torch.float)
    reco_score_n = torch.zeros([reco.shape[0], 20], dtype=torch.float)
    for u in range(reco.shape[0]):
        for i in range(reco.shape[1]):
            reco_score_n[u][i] = r_hat_n[u][reco[u][i]]
            reco_score_p[u][i] = r_hat_p[u][reco[u][i]]

    _, reco_n = torch.topk(reco_score_n, 20)
    reco_final = torch.zeros([reco.shape[0], K], dtype=torch.int32)
    for u in range(reco_score_p.shape[0]):
        for i in range(10):
            reco_final[u, i] = reco[u, reco_n[u][19 - i]]
    reco_final = reco_final.numpy()

    return reco_final


def gen_top_k_new3(data_class, r_hat_p, r_hat_n, K=300):
    all_items = set(np.arange(1, data_class.num_v + 1))
    tot_items = set(data_class.train['movieId']).union(set(data_class.test['movieId']))
    no_items = all_items - tot_items
    tot_items = torch.tensor(list(tot_items)) - 1
    no_items = (torch.tensor(list(no_items)) - 1).long()
    r_hat_p[:, no_items] = -np.inf
    for u, i in data_class.train.values[:, :-1] - 1:
        r_hat_p[u, i] = -np.inf

    _, reco_n = torch.topk(r_hat_n, K)

    threshold = 0.0

    thred_array = np.where(r_hat_n > threshold)
    for i in range(len(thred_array[0])):
        r_hat_p[thred_array[0][i], thred_array[1][i]] = -np.inf

    _, reco = torch.topk(r_hat_p, K)
    reco = reco.numpy()

    return reco


def get_top_k(data_class, r_hat, r_hat_n, K=300):
    all_items = set(np.arange(1, data_class.num_v + 1))
    tot_items = set(data_class.train['movieId']).union(set(data_class.test['movieId']))
    no_items = all_items - tot_items
    tot_items = torch.tensor(list(tot_items)) - 1
    no_items = (torch.tensor(list(no_items)) - 1).long()
    r_hat[:, no_items] = -np.inf
    for u, i in data_class.train.values[:, :-1] - 1:
        r_hat[u, i] = -np.inf

    _, reco_n = torch.topk(r_hat_n, 20)
    for u in range(reco_n.shape[0]):
        for i in reco_n[u]:
            r_hat[u, i] = -np.inf

    _, reco = torch.topk(r_hat, K)
    reco = reco.numpy()

    return reco
