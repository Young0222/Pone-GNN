from copy import deepcopy

import numpy as np
import torch


class evaluator():
    def __init__(self, data_class, reco, args, N=None, partition=None):
        if partition is None:
            partition = [20, 50]
        if N is None:
            N = [1, 5, 10, 15, 20]
        print('*** evaluation phase ***')

        self.reco = reco
        self.data = data_class
        self.N = np.array(N)
        self.threshold = 3  # 0 to test baselines  # to generate ground truth set; for ML-1M, yelp, amazon-book
        # self.threshold = 0.5 # to generate ground truth set; for kuai_rec
        self.partition = partition

        all_items = set(np.arange(1, data_class.num_v + 1))
        tot_items = set(data_class.train['movieId']).union(set(data_class.test['movieId']))
        no_items = all_items - tot_items
        tot_items = torch.tensor(list(tot_items)) - 1
        self.no_item = (torch.tensor(list(no_items)) - 1).long().numpy()

        self.__gen_ground_truth_set()
        self.__group_partition()

    def __gen_ground_truth_set(self):
        print('*** ground truth set ***')
        self.GT = dict()
        self.GT2 = dict()
        temp = deepcopy(self.data.test)
        temp = temp[temp['rating'] >= self.threshold].values[:, :-1] - 1
        temp2 = deepcopy(self.data.test)
        temp2 = temp2[temp2['rating'] < self.threshold].values[:, :-1] - 1
        for j in range(self.data.num_u):
            if len(temp[temp[:, 0] == j][:, 1]) > 0:
                self.GT[j] = temp[temp[:, 0] == j][:, 1]
            if len(temp2[temp2[:, 0] == j][:, 1]) > 0:
                self.GT2[j] = temp2[temp2[:, 0] == j][:, 1]

    def __group_partition(self):
        print('*** ground partition ***')
        unique_u, counts_u = np.unique(self.data.train['userId'].values - 1, return_counts=True)
        self.G = dict()
        self.G['group1'] = unique_u[np.argwhere(counts_u < self.partition[0])].reshape(-1)
        temp = unique_u[np.argwhere(counts_u < self.partition[1])]
        self.G['group2'] = np.setdiff1d(temp, self.G['group1'])
        self.G['group3'] = np.setdiff1d(unique_u, temp)
        self.G['total'] = unique_u

    def precision_and_recall(self):
        print('*** precision ***')
        self.p = dict()
        self.r = dict()
        self.h = dict()
        self.tp = dict()
        self.fp = dict()
        self.fn = dict()
        leng = dict()
        maxn = max(self.N)
        for i in [j for j in self.G]:
            self.p[i] = np.zeros(maxn)
            self.r[i] = np.zeros(maxn)
            self.h[i] = np.zeros(maxn)
            self.tp[i] = np.zeros(maxn)
            self.fp[i] = np.zeros(maxn)
            self.fn[i] = np.zeros(maxn)
            leng[i] = 0
        for uid in [j for j in self.GT]:
            leng['total'] += 1
            # import IPython; IPython.embed()
            hit_ = np.cumsum([1.0 if item in self.GT[uid] else 0.0 for idx, item in enumerate(self.reco[uid][:maxn])])
            self.p['total'] += hit_ / np.arange(1, maxn + 1)  # 得到当前user的precision,累加全部user的precision,得到p['total']
            self.r['total'] += hit_ / len(self.GT[uid])
            new_hit_ = deepcopy(hit_)
            new_hit_[new_hit_ >= 1] = 1
            self.h['total'] += new_hit_
            self.tp['total'] += hit_
            self.fp['total'] += (np.arange(1, maxn + 1) - hit_)
            self.fn['total'] += (len(self.GT[uid]) - hit_)
            if uid in self.G['group1']:
                self.p['group1'] += hit_ / np.arange(1, maxn + 1)
                self.r['group1'] += hit_ / len(self.GT[uid])
                leng['group1'] += 1
            elif uid in self.G['group2']:
                self.p['group2'] += hit_ / np.arange(1, maxn + 1)
                self.r['group2'] += hit_ / len(self.GT[uid])
                leng['group2'] += 1
            elif uid in self.G['group3']:
                self.p['group3'] += hit_ / np.arange(1, maxn + 1)
                self.r['group3'] += hit_ / len(self.GT[uid])
                leng['group3'] += 1
        for i in [j for j in self.G]:  # 对用户求平均
            self.p[i] /= leng[i]
            self.r[i] /= leng[i]
            self.h[i] /= leng[i]
            self.tp[i] /= leng[i]
            self.fp[i] /= leng[i]
            self.fn[i] /= leng[i]

    def calculate_neg_sim(self, r_hat_n):
        print('*** results of similarity scores of true negatives ***')
        self.neg_res = []
        for uid in [j for j in self.GT2]:
            for item in self.GT2[uid]:
                self.neg_res.append(r_hat_n[uid][item])
        self.neg_res = np.array(self.neg_res)
        self.total_num_n = len(self.neg_res)
        self.cal_num_9_n = len(np.where(self.neg_res > 0.9)[0])
        self.cal_num_8_n = len(np.where(self.neg_res > 0.8)[0])
        self.cal_num_7_n = len(np.where(self.neg_res > 0.7)[0])
        self.cal_num_6_n = len(np.where(self.neg_res > 0.6)[0])
        self.cal_num_5_n = len(np.where(self.neg_res > 0.5)[0])
        self.cal_num_4_n = len(np.where(self.neg_res > 0.4)[0])
        self.cal_num_3_n = len(np.where(self.neg_res > 0.3)[0])
        self.cal_num_2_n = len(np.where(self.neg_res > 0.2)[0])
        self.cal_num_1_n = len(np.where(self.neg_res > 0.1)[0])
        self.cal_num_0_n = len(np.where(self.neg_res > 0.0)[0])
        self.neg_mean = np.mean(self.neg_res)

    def calculate_pos_sim(self, r_hat_p):
        print('*** results of similarity scores of true negatives ***')
        self.pos_res = []
        for uid in [j for j in self.GT]:
            for item in self.GT[uid]:
                self.pos_res.append(r_hat_p[uid][item])
        self.pos_res = np.array(self.pos_res)
        self.total_num_p = len(self.pos_res)
        self.cal_num_9_p = len(np.where(self.pos_res > 0.9)[0])
        self.cal_num_8_p = len(np.where(self.pos_res > 0.8)[0])
        self.cal_num_7_p = len(np.where(self.pos_res > 0.7)[0])
        self.cal_num_6_p = len(np.where(self.pos_res > 0.6)[0])
        self.cal_num_5_p = len(np.where(self.pos_res > 0.5)[0])
        self.cal_num_4_p = len(np.where(self.pos_res > 0.4)[0])
        self.cal_num_3_p = len(np.where(self.pos_res > 0.3)[0])
        self.cal_num_2_p = len(np.where(self.pos_res > 0.2)[0])
        self.cal_num_1_p = len(np.where(self.pos_res > 0.1)[0])
        self.cal_num_0_p = len(np.where(self.pos_res > 0.0)[0])
        self.pos_mean = np.mean(self.pos_res)

    def normalized_DCG(self):
        print('*** nDCG ***')
        self.nDCG = dict();
        leng = dict()
        maxn = max(self.N)

        for i in [j for j in self.G]:
            self.nDCG[i] = np.zeros(maxn)
            leng[i] = 0
        for uid in [j for j in self.GT]:
            leng['total'] += 1
            idcg_len = min(len(self.GT[uid]), maxn)
            temp_idcg = np.cumsum(1.0 / np.log2(np.arange(2, maxn + 2)))
            temp_idcg[idcg_len:] = temp_idcg[idcg_len - 1]
            temp_dcg = np.cumsum([1.0 / np.log2(idx + 2) if item in self.GT[uid] else 0.0 for idx, item in
                                  enumerate(self.reco[uid][:maxn])])
            self.nDCG['total'] += temp_dcg / temp_idcg
            if uid in self.G['group1']:
                self.nDCG['group1'] += temp_dcg / temp_idcg
                leng['group1'] += 1
            elif uid in self.G['group2']:
                self.nDCG['group2'] += temp_dcg / temp_idcg
                leng['group2'] += 1
            elif uid in self.G['group3']:
                self.nDCG['group3'] += temp_dcg / temp_idcg
                leng['group3'] += 1
        for i in [j for j in self.G]:
            self.nDCG[i] /= leng[i];
