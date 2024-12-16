import torch
import torch.nn as nn
import torch.nn.functional as F

from convols import LightSignedConv, LightSignedConv2, LightGINConv2


class PoneGNN(nn.Module):
    def __init__(self, num_u, num_v, num_layer=2, dim=64, reg=1e-4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.M = num_u  # user number
        self.N = num_v  # item number
        self.num_layer = num_layer
        self.dim = dim
        self.reg = reg
        self.embed_dim = dim
        self.temperature = 1.0
        self.contrastive_weight = 1.0

        self.user_embedding = nn.Parameter(torch.empty(self.M, self.dim))
        self.item_embedding = nn.Parameter(torch.empty(self.N, self.dim))
        nn.init.xavier_normal_(self.user_embedding)
        nn.init.xavier_normal_(self.item_embedding)
        self.user_neg_embedding = nn.Parameter(torch.empty(self.M, self.dim))
        self.item_neg_embedding = nn.Parameter(torch.empty(self.N, self.dim))
        nn.init.xavier_normal_(self.user_neg_embedding)
        nn.init.xavier_normal_(self.item_neg_embedding)

        self.conv = nn.ModuleList()
        hidden_dim = self.dim
        for i in range(self.num_layer):
            if i == 0:
                self.conv.append(LightGINConv2(hidden_dim, hidden_dim, True))
            else:
                self.conv.append(LightGINConv2(hidden_dim, hidden_dim, False))


    def forward(self, data_p, data_n):
        pos_edges = data_p.edge_index
        neg_edges = data_n.edge_index
        alpha = 1. / (self.num_layer + 1)

        ego_pos_embeddings = torch.cat((self.user_embedding, self.item_embedding), dim=0)
        ego_neg_embeddings = torch.cat((self.user_neg_embedding, self.item_neg_embedding), dim=0)
        ego_embeddings = (ego_pos_embeddings, ego_pos_embeddings)
        pos_embeddings, neg_embeddings = ego_pos_embeddings * alpha, ego_neg_embeddings * alpha

        for i in range(self.num_layer):
            ego_embeddings = self.conv[i](ego_embeddings, pos_edges, neg_edges)
            pos_embeddings, neg_embeddings = pos_embeddings + ego_embeddings[0] * alpha, neg_embeddings + ego_embeddings[1] * alpha
        return pos_embeddings, neg_embeddings


    def loss(self, users, items, weights, negative_samples, data_p, data_n, EPOCH):
        """
        Args:
            users: batch users id
            items: batch items id
            weights: user-item (ratings-offset)
            negative_samples: negative samples for BPR
            data_p: positive edges
            data_n: negative edges

        Returns: loss to backward

        """
        loss = 0.
        pos_emb, neg_emb = self(data_p, data_n)

        # 1. Positive BPR Loss
        u_p = pos_emb[users]
        i_p = pos_emb[items]
        n_p = pos_emb[negative_samples]
        positive_batch = torch.mul(u_p, i_p)
        negative_batch = torch.mul(u_p.view(len(u_p), 1, self.embed_dim), n_p)
        pos_bpr_loss = F.logsigmoid(
            (-1/2*torch.sign(weights) + 3/2).view(len(u_p), 1) * positive_batch.sum(dim=1).view(len(u_p), 1)
            - negative_batch.sum(dim=2)
        ).sum(dim=1)
        pos_bpr_loss = torch.mean(pos_bpr_loss)
        loss += -pos_bpr_loss
        reg_loss_1 = 1. / 2 * (u_p ** 2).sum() + 1. / 2 * (i_p ** 2).sum() + 1. / 2 * (n_p ** 2).sum()
        loss += self.reg * reg_loss_1
        
        if EPOCH % 10 == 1: # Trigger learning
            # 2. Negative BPR Loss
            u_n = neg_emb[users]
            i_n = neg_emb[items]
            n_n = neg_emb[negative_samples]
            positive_batch = torch.mul(u_n, i_n)
            negative_batch = torch.mul(u_n.view(len(u_p), 1, self.embed_dim), n_n)
            neg_bpr_loss = F.logsigmoid(
                negative_batch.sum(dim=2)
                - (1/2*torch.sign(weights) + 3/2).view(len(u_n), 1) * positive_batch.sum(dim=1).view(len(u_n),1)
            ).sum(dim=1)
            neg_bpr_loss = torch.mean(neg_bpr_loss)
            loss += -neg_bpr_loss
            reg_loss_2 = 1. / 2 * (u_n ** 2).sum() + 1. / 2 * (i_n ** 2).sum() + 1. / 2 * (n_n ** 2).sum()
            loss += self.reg * reg_loss_2

            # 3. Contrastive loss
            u_p_norm, i_p_norm, n_p_norm = F.normalize(u_p, dim=1), F.normalize(i_p, dim=1), F.normalize(n_p, dim=1)
            u_n_norm, i_n_norm, n_n_norm = F.normalize(u_n, dim=1), F.normalize(i_n, dim=1), F.normalize(n_n, dim=1)
            positive_similarity, negative_similarity = torch.sum(torch.mul(u_p_norm, i_p_norm), dim=1), torch.sum(torch.mul(u_n_norm, i_n_norm), dim=1)
            positive_pair_similarity, negative_pair_similarity = torch.exp(positive_similarity / self.temperature), torch.exp(negative_similarity / self.temperature)
            contrastive_loss = -torch.log(positive_pair_similarity / (positive_pair_similarity + negative_pair_similarity)).mean()
            loss += self.contrastive_weight * contrastive_loss

        return loss

    @torch.no_grad()
    def get_ui_embeddings(self, data_p, data_n):
        pos_embeddings, neg_embeddings = self(data_p, data_n)
        u_p_embeddings, i_p_embeddings = torch.split(pos_embeddings, [self.M, self.N], dim=0)
        u_n_embeddings, i_n_embeddings = torch.split(neg_embeddings, [self.M, self.N], dim=0)
        return u_p_embeddings, u_n_embeddings, i_p_embeddings, i_n_embeddings






