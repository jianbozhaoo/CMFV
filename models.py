import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, ELU, LeakyReLU, Sigmoid
import random
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import heapq
from tqdm import tqdm
from sklearn.cluster import KMeans
from pytorch_pretrained_bert.modeling import BertModel
eps = 1e-12

class SelfAttention(nn.Module):
    def __init__(self, nhid):
        super(SelfAttention, self).__init__()
        self.nhid = nhid
        self.project = nn.Sequential(
            Linear(nhid, 64),
            ELU(),
            Linear(64, 1),
            ELU(),
        )
    def forward(self, evidences, claims, evi_labels=None):  
        # evidences [256,5,768] claims [256,768] evi_labels [256,5]
        # claims = claims.unsqueeze(1).repeat(1,evidences.shape[1],1)  # [256,5,768]
        claims = claims.unsqueeze(1).expand(claims.shape[0],evidences.shape[1],claims.shape[-1])  # [256,5,768]
        temp = torch.cat((claims,evidences),dim=-1)  # [256,5,768*2]
        weight = self.project(temp)  # [256,5,1]
        if evi_labels is not None:
            # evi_labels = evi_labels[:,1:] # [batch,5]
            mask = evi_labels == 0 # [batch,5]
            mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float("-inf")) #  [batch,5]
            weight = weight + mask.unsqueeze(-1) # [256,5,1]
        weight = F.softmax(weight,dim=1)  # [256,5,1]
        outputs = torch.matmul(weight.transpose(1,2), evidences).squeeze(dim=1)  # [256,768]
        return outputs
    
class CrossAttention(nn.Module):
    def __init__(self, nhid):
        super(CrossAttention, self).__init__()
        self.project_c = Linear(nhid, 64)
        self.project_e = Linear(nhid, 64)

        self.f_align = Linear(4*nhid,nhid)
    def forward(self, x, datas):
        batch = datas.batch
        claim_batch = batch[datas.claim_index] # [500]
        evidence_batch = batch[datas.evidence_index] # [1000]

        mask = ~(claim_batch.unsqueeze(1) == evidence_batch.unsqueeze(0))  # [500,1000]
        mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float("-inf")) # [500,1000]
        claim = x[datas.claim_index] # [500,768]
        evidence = x[datas.evidence_index] # [1000,768]
        weight_c = self.project_c(claim)  # [500,64]
        weight_e = self.project_e(evidence)  # [1000,64]
        weight = torch.matmul(weight_c, weight_e.transpose(0,1)) # [500,1000]
        weight = weight + mask
        weight = F.softmax(weight,dim=-1) # [500,1000]
        claim_new = torch.matmul(weight,evidence) # [500,768]

        a = torch.cat([claim,claim_new,claim-claim_new,claim*claim_new],dim=-1) # [500,768*4]
        a = self.f_align(a) # [500,768]
        res = global_mean_pool(a, claim_batch) # [128,768]
        
        return res
    
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        self.alpha = self.alpha.to(target.device)
        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class ONE_ATTENTION_with_bert(torch.nn.Module):
    def __init__(self, nfeat, nclass, evi_max_num) -> None:
        super(ONE_ATTENTION_with_bert, self).__init__()
        self.evi_max_num = evi_max_num
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.conv1 = GCNConv(nfeat, nfeat)
        self.conv2 = GCNConv(nfeat, nfeat)
        self.attention = SelfAttention(nfeat*2)
        self.classifier = nn.Sequential(
            Linear(nfeat , nfeat),
            ELU(True),
            Linear(nfeat, nclass),
            ELU(True),
        )

    def cal_graph_representation(self, data):
        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data
        input_ids = input_ids.view(-1,input_ids.shape[-1])
        input_mask = input_mask.view(-1,input_ids.shape[-1])
        segment_ids = segment_ids.view(-1,input_ids.shape[-1])
        _, pooled_output = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,)
        pooled_output = pooled_output.view(-1,1+self.evi_max_num,pooled_output.shape[-1]) # [batch,6,768]
        datas = []
        for i in range(len(pooled_output)):
            x = pooled_output[i] # [6,768]

            edge_index = torch.arange(sent_labels[i].sum().item())
            edge_index = torch.cat([edge_index.unsqueeze(0).repeat(1,sent_labels[i].sum().item()),
                                    edge_index.unsqueeze(1).repeat(1,sent_labels[i].sum().item()).view(1,-1)],dim=0) # [2,36]
            edge_index1 = torch.cat([edge_index[1].unsqueeze(0),edge_index[0].unsqueeze(0)],dim=0)
            edge_index = torch.cat([edge_index,edge_index1],dim=1)
            edge_index = edge_index.to(x.device)
            data = Data(x=x, edge_index=edge_index)
            data.validate(raise_on_error=True)
            datas.append(data)
        datas = Batch.from_data_list(datas)
        x, edge_index = datas.x, datas.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.normalize(x,dim=-1)
        x = x.view(-1,1+self.evi_max_num,x.shape[-1]) # [batch,6,768]
        feature_batch, claim_batch = x[:,1:,:], x[:,0,:] # [batch,5,768] # [batch,768]
        graph_rep = self.attention(feature_batch, claim_batch, sent_labels[:,1:]) # [batch,768]
        return graph_rep

    def forward(self, data):
        graph_rep = self.cal_graph_representation(data)
        outputs = self.classifier(graph_rep)
        return outputs

class CMFV(nn.Module):
    def __init__(self, nfeat, nclass, max_length, beam_size, max_evi_num, lambda_val, causal_method):
        super(CMFV, self).__init__()
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.max_length = max_length
        self.beam_size = beam_size
        self.max_evi_num = max_evi_num
        self.causal_method = causal_method
        self.conv1 = GCNConv(nfeat, nfeat)
        self.conv2 = GCNConv(nfeat, nfeat)
        self.nclass = nclass
        self.lambda_val = lambda_val

        self.mlp1 = nn.Sequential(
            Linear(3*nfeat, nfeat),
            ELU(),
            Linear(nfeat, 1),
            ELU(),
        )
        
        self.attention = SelfAttention(nfeat*2)

        if "cf" in self.causal_method:
            self.classifier_claim = nn.Sequential(
                Linear(768 , self.nclass),
                ELU(),
            )
            self.constant = nn.Parameter(torch.tensor(0.0))
        
        self.lstm = nn.LSTM(nfeat,nfeat,2,batch_first=True)

        self.classifier = nn.Sequential(
            Linear(nfeat , nfeat),
            ELU(),
            Linear(nfeat , nclass),
            ELU(),
        )
        for m in self.mlp1.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        for m in self.classifier.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)


    def paths_to_class_logit(self,x,paths,logits,paths_mask,evi_labels): 
        # [batch,6,768] [batch,5,4] [batch,5] [batch,5,4] [batch,6]
        evidences, claims = x[:,1:,:], x[:,0,:] # [batch,5,768] # [batch,768]
        graph_rep = self.attention(evidences, claims, evi_labels[:,1:]) # [batch,768]

        paths = torch.tensor(paths, dtype=torch.long, device=x.device)
        paths_mask = torch.tensor(paths_mask, dtype=torch.long, device=x.device)
        logits = [[j.unsqueeze(0) for j in i] for i in logits]
        logits = [torch.cat(i,dim=0) for i in logits]
        logits = torch.cat([i.unsqueeze(0) for i in logits],dim=0) # [batch,5]
        # logits = torch.tensor(logits, device=x.device) # [batch,5]
        logits = F.softmax(logits,dim=1) # [batch,5]
        paths_rep = []
        for i in range(len(x)):
            graph = x[i] # [6,768]
            rep = graph[paths[i]] # [5,4,768]
            paths_rep.append(rep.unsqueeze(0))
        paths_rep = torch.cat(paths_rep,dim=0) # [batch,5,4,768]
        paths_rep = paths_rep * paths_mask.unsqueeze(-1) # [batch,5,4,768]
        shape = paths_rep.shape # [batch,5,4,768]
        paths_rep = paths_rep.view(shape[0]*shape[1],shape[2],shape[3]).contiguous() # [batch*5,4,768]

        graph_rep = graph_rep.unsqueeze(1).expand(-1,shape[1],-1) # [batch,5,768]
        graph_rep = graph_rep.contiguous().view(shape[0]*shape[1],-1) # [batch*5,768]
        graph_rep = graph_rep.unsqueeze(0).expand(2,-1,-1).contiguous() # [2,batch*5,768]

        h0, c0 = graph_rep, graph_rep
        output, (hn, cn) = self.lstm(paths_rep,(h0,c0)) # [batch*5,4,768] [1,batch*5,768]
        # output, (hn, cn) = self.lstm(paths_rep) # [batch*5,4,768] [1,batch*5,768]
        paths_rep = output[:,-1,:] # [batch*5,768]
        paths_rep = paths_rep.view(shape[0],shape[1],-1) # [batch,5,768]

        paths_class = self.classifier(paths_rep) # [batch,5,3]
        paths_class = F.softmax(paths_class,dim=-1) # [batch,5,3]
        paths_class = paths_class * logits.unsqueeze(-1) # [batch,5,3]
        paths_class = paths_class.sum(dim=1) # [batch,3]
        return paths_class # [batch,3]   
    
    def claim_classifier(self,claim_batch,):  # claim_batch [batch,768]
        res = self.classifier_claim(claim_batch) # [batch,3]
        return res # [10,3]

    def counterfactual_reasoning(self, res_direct, res_claim, res, lambda_val):
        constant = torch.sigmoid(self.constant)
        te_fusion = torch.log(1e-9 + torch.sigmoid(res_direct + res_claim + res))
        nde_final = torch.log(1e-9 + torch.sigmoid(res_direct.detach() + constant * torch.ones_like(res_claim) + constant * torch.ones_like(res)))
        nde_claim = torch.log(1e-9 + torch.sigmoid(res_claim.detach() + constant * torch.ones_like(res_direct) + constant * torch.ones_like(res)))
        tie = te_fusion - lambda_val * nde_claim - lambda_val * nde_final
        return tie, te_fusion, nde_final, nde_claim

    
    def forward(self, data, centers=None, evi_supervision=False):
        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data
        input_ids = input_ids.view(-1,input_ids.shape[-1])
        input_mask = input_mask.view(-1,input_ids.shape[-1])
        segment_ids = segment_ids.view(-1,input_ids.shape[-1])
        _, pooled_output = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,)
        pooled_output = pooled_output.view(-1,1+self.max_evi_num,pooled_output.shape[-1]) # [batch,6,768]

        feature_batch, claim_batch = pooled_output[:,1:,:], pooled_output[:,0,:] # [batch,5,768] # [batch,768]
        
        direct = self.attention(feature_batch, claim_batch, evi_labels[:,1:])  # [batch, nfeat]
        res_direct = self.classifier(direct)  # [batch, nclass]

        datas = []
        for i in range(len(feature_batch)):
            x = torch.cat([claim_batch[i].unsqueeze(0),
                           feature_batch[i]],dim=0) # [6,768]

            edge_index = torch.arange(sent_labels[i].sum().item())
            edge_index = torch.cat([edge_index.unsqueeze(0).repeat(1,sent_labels[i].sum().item()),
                                    edge_index.unsqueeze(1).repeat(1,sent_labels[i].sum().item()).view(1,-1)],dim=0) # [2,36]
            edge_index1 = torch.cat([edge_index[1].unsqueeze(0),edge_index[0].unsqueeze(0)],dim=0)
            edge_index = torch.cat([edge_index,edge_index1],dim=1)
            edge_index = edge_index.to(x.device)
            data = Data(x=x, edge_index=edge_index)
            # data.validate(raise_on_error=True)
            datas.append(data)
        datas = Batch.from_data_list(datas)
        
        x, edge_index = datas.x, datas.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.normalize(x,dim=-1)

        x = x.view(-1,1+self.max_evi_num,x.shape[-1]) # [batch,6,768]
        # x = torch.cat([claim_batch.unsqueeze(1),feature_batch],dim=1) # [batch,6,768]
        probability = self.cal_transition_probability_matrix(x,sent_labels)  # [batch,6,6]

        paths, logits, paths_mask = self.batch_find_path(probability, self.max_length, self.beam_size)
        res = self.paths_to_class_logit(x,paths,logits,paths_mask,sent_labels)
        self.causal_method == "cf"
        res_claim = self.claim_classifier(claim_batch)  #[batch,3]
        lambda_val = self.lambda_val
        tie, te_fusion, nde_final, nde_claim = self.counterfactual_reasoning(res_direct, res_claim, res, lambda_val)
        return tie, te_fusion, nde_final, nde_claim, res
    

    def cal_transition_probability_matrix(self, x, evi_labels): # [batch,6,768] [batch,6]
        node_num = x.shape[1]
        mat_rep = torch.cat([x.unsqueeze(2).expand(-1,-1,node_num,-1), #[batch,6,6,768]
                             x.unsqueeze(1).expand(-1,node_num,-1,-1), #[batch,6,6,768]
                             x[:,0,:].unsqueeze(1).unsqueeze(1).expand(-1,node_num,node_num,-1), #[batch,6,6,768]
                             ],dim=-1) # [batch,6,6,768*3]
        weight = self.mlp1(mat_rep) # [batch,6,6,1]
        weight = weight.squeeze(-1) # [batch,6,6]
        
        # weight = torch.matmul(x,x.transpose(1,2)) # [batch,6,6]
        # print(weight.sum().item())
        evi_labels = evi_labels.to(torch.float)
        mask = torch.matmul(evi_labels.unsqueeze(2),evi_labels.unsqueeze(1)) # [batch,6,6]
        mask = mask + torch.diag_embed(torch.ones(mask.shape[1])).to(mask.device)
        mask = mask == 0
        mask = torch.zeros_like(mask,dtype=torch.float32).masked_fill(mask,float("-inf")) #  [batch,6,6]
        weight = weight + mask
        probability = F.softmax(weight,dim=-1) # [batch,6,6]
        return probability # [batch,6,6]
    

    def find_path(self, start_node, probability, max_length, beam_size): # 0 [6,6] 4 5
        top_beam_paths = [[start_node]]
        top_beam_logits = [torch.tensor(0.0).to(probability.device)]
        top_beam_end = [0]
        while not all(top_beam_end):
            new_paths = []
            new_logits = []
            for k in range(len(top_beam_paths)):
                if top_beam_end[k]:
                    continue
                path = top_beam_paths[k]
                logit = top_beam_logits[k]
                curr_node = path[-1]
                edge = probability[curr_node] # [6]

                next_nodes = []
                values = []
                for i, value in enumerate(edge):
                    if i not in path and value > 0.0:
                        next_nodes.append(i)
                        values.append(value)
                top_index = heapq.nlargest(min(beam_size,len(values)), range(len(values)), values.__getitem__)
                next_nodes = [next_nodes[j] for j in top_index] 
                values = [values[j] for j in top_index]
                
                new_paths += [path + [j] for j in next_nodes]
                # new_logits += [logit + torch.log(torch.tensor(j + 1e-8).to(probability.device)) for j in values]
                values = [torch.tensor(v, dtype=torch.float32, device=probability.device) for v in values]
                new_logits += [logit + torch.log(v + 1e-8) for v in values]


            all_paths = []
            all_logits = []
            for index, end in enumerate(top_beam_end):
                if end:
                    all_paths.append(top_beam_paths[index])
                    all_logits.append(top_beam_logits[index])
            all_paths += new_paths
            all_logits += new_logits

            # 取top beam_size个
            temp_logits = [logit/len(all_paths[index]) for index, logit in enumerate(all_logits)]
            top_index = heapq.nlargest(min(beam_size,len(temp_logits)), range(len(temp_logits)), temp_logits.__getitem__)
            top_beam_paths = [all_paths[j] for j in top_index] 
            top_beam_logits = [all_logits[j] for j in top_index]

            top_beam_end = []
            for i in range(len(top_beam_paths)):
                end = 0
                if len(top_beam_paths[i]) >= max_length:
                    end = 1
                curr_node = top_beam_paths[i][-1]
                edge = probability[curr_node]
                next_nodes = []
                for j, value in enumerate(edge):
                    if j not in path and value > 0.0:
                        next_nodes.append(j)
                top_index = heapq.nlargest(min(beam_size,len(values)), range(len(values)), values.__getitem__)
                next_nodes = [next_nodes[j] for j in top_index] 
                if all([node in top_beam_paths[i] for node in next_nodes]) or len(next_nodes)==0:
                    end = 1
                top_beam_end.append(end)

        top_beam_logits = [logit/len(top_beam_paths[index]) for index, logit in enumerate(top_beam_logits)]
        paths = top_beam_paths
        logits = top_beam_logits
        return paths, logits # [5,4] [5]


    
    def batch_find_path(self, probability, max_length, beam_size): # [batch,6,6]
        all_paths = [] # [batch,5,4]
        all_paths_mask = [] # [batch,5,4]
        all_logits = [] # [batch,5]
        for i in range(len(probability)):
            paths, logits = self.find_path(0,probability[i],max_length,beam_size)
            paths_mask = []
            for j in range(len(paths)):

                paths_mask.append([1]*len(paths[j])+[0]*(max_length - len(paths[j])))
                paths[j] = paths[j] + [0] * (max_length - len(paths[j]))

            paths_mask += [[0]*max_length]*(beam_size-len(paths))
            logits += [torch.tensor(float("-inf")).to(probability.device)]*(beam_size-len(paths))
            paths += [[0]*max_length]*(beam_size-len(paths))
            
            all_paths.append(paths)
            all_paths_mask.append(paths_mask)
            all_logits.append(logits)

        device = probability.device
        paths_tensor = torch.tensor(all_paths, device=device)          
        logits_tensor = torch.stack([torch.stack(l) for l in all_logits], dim=0).to(device)
        paths_mask_tensor = torch.tensor(all_paths_mask, device=device)  
        return paths_tensor, logits_tensor, paths_mask_tensor # [batch,3,5] [batch,3] [batch,3,5]


class CLASSIFIER(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CLASSIFIER, self).__init__()
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.mlp = nn.Sequential(
            Linear(nfeat, nclass),
            ELU(True),
        )

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data
        _, pooled_output = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,)
        res = self.mlp(pooled_output)
        return res

class CLEVER(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CLEVER, self).__init__()
        self.bert1 = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.bert2 = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.mlp1 = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
            # Sigmoid(),
        )
        self.mlp2 = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
            # Sigmoid(),
        )
        self.constant = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,0,:] # [batch,128]
        input_mask = input_mask[:,0,:] # [batch,128]
        segment_ids = segment_ids[:,0,:] # [batch,128]
        _, claims = self.bert1(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,1,:] # [batch,128]
        input_mask = input_mask[:,1,:] # [batch,128]
        segment_ids = segment_ids[:,1,:] # [batch,128]
        _, evidences = self.bert2(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        # res_claim = torch.log(1e-8 + self.mlp1(claims)) # [batch,3]
        # res_fusion = torch.log(1e-8 + self.mlp2(evidences)) # [batch,3]
        res_claim = self.mlp1(claims) # [batch,3]
        res_fusion = self.mlp2(evidences) # [batch,3]
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_fusion))
        cf_res = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_fusion)))

        tie = res_final - cf_res
        
        # res_final = res_claim + res_fusion
        # cf_res = res_claim.detach() + self.constant * torch.ones_like(res_fusion)
        # tie = res_final - cf_res
        # tie = res_fusion - res_claim
        # return res_claim, res_final, cf_res, tie
        return res_claim, res_final, cf_res, tie


class CICR(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CICR, self).__init__()
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.classifier_claim = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        self.classifier_evidence = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        # constant_evidence = torch.nn.Parameter(torch.zeros((nfeat)))
        self.classifier_fusion = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        # constant_fusion = torch.nn.Parameter(torch.zeros((nfeat)))
        self.constant = nn.Parameter(torch.tensor(0.0))
        # self.linear1 = Linear(nclass, nclass)
        # self.linear2 = Linear(nclass, nclass)
      

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,0,:] # [batch,128]
        input_mask = input_mask[:,0,:] # [batch,128]
        segment_ids = segment_ids[:,0,:] # [batch,128]
        _, claims = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,1,:] # [batch,128]
        input_mask = input_mask[:,1,:] # [batch,128]
        segment_ids = segment_ids[:,1,:] # [batch,128]
        _, claim_evidences = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]

        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,2,:] # [batch,128]
        input_mask = input_mask[:,2,:] # [batch,128]
        segment_ids = segment_ids[:,2,:] # [batch,128]
        _, evidences = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        claims = claims.detach()
        evidences = evidences.detach()

        res_claim = self.classifier_claim(claims) # [batch,3]
        res_evidence = self.classifier_evidence(evidences) # [batch,3]
        res_fusion = self.classifier_fusion(claim_evidences) # [batch,3]
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_evidence + res_fusion))

        counterfactual_final = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_evidence) \
             + self.constant * torch.ones_like(res_fusion)))
        TIE = res_final - counterfactual_final

        return res_claim, res_evidence, res_final, counterfactual_final, TIE


class CLEVER(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CLEVER, self).__init__()
        self.bert1 = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.bert2 = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.mlp1 = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
            # Sigmoid(),
        )
        self.mlp2 = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
            # Sigmoid(),
        )
        self.constant = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,0,:] # [batch,128]
        input_mask = input_mask[:,0,:] # [batch,128]
        segment_ids = segment_ids[:,0,:] # [batch,128]
        _, claims = self.bert1(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,1,:] # [batch,128]
        input_mask = input_mask[:,1,:] # [batch,128]
        segment_ids = segment_ids[:,1,:] # [batch,128]
        _, evidences = self.bert2(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        # res_claim = torch.log(1e-8 + self.mlp1(claims)) # [batch,3]
        # res_fusion = torch.log(1e-8 + self.mlp2(evidences)) # [batch,3]
        res_claim = self.mlp1(claims) # [batch,3]
        res_fusion = self.mlp2(evidences) # [batch,3]
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_fusion))
        cf_res = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_fusion)))

        tie = res_final - cf_res
        
        # res_final = res_claim + res_fusion
        # cf_res = res_claim.detach() + self.constant * torch.ones_like(res_fusion)
        # tie = res_final - cf_res
        # tie = res_fusion - res_claim
        # return res_claim, res_final, cf_res, tie
        return res_claim, res_final, cf_res, tie

class CLEVER_graph(nn.Module):
    def __init__(self, nfeat, nclass, evi_max_num):
        super(CLEVER_graph, self).__init__()
        self.evi_max_num = evi_max_num
        self.fusion_model = ONE_ATTENTION_with_bert(nfeat, nclass, evi_max_num)
        self.claim_model = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.mlp_claim = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        self.constant = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        res_fusion = self.fusion_model(data) # [batch,3]

        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels, indexs = data
        input_ids = input_ids[:,0,:] # [batch,128]
        input_mask = input_mask[:,0,:] # [batch,128]
        segment_ids = segment_ids[:,0,:] # [batch,128]
        _, claims = self.claim_model(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        res_claim = self.mlp_claim(claims) # [batch,3]
   
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_fusion))
        cf_res = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_fusion)))

        tie = res_final - cf_res
        
        # res_final = res_claim + res_fusion
        # cf_res = res_claim.detach() + self.constant * torch.ones_like(res_fusion)
        # tie = res_final - cf_res
        # tie = res_fusion - res_claim
        # return res_claim, res_final, cf_res, tie
        return res_claim, res_final, cf_res, tie

class ONE_ATTENTION(nn.Module):
    def __init__(self, nfeat, nclass, evi_max_num, pool):
        super(ONE_ATTENTION, self).__init__()
        self.evi_max_num = evi_max_num
        self.pool = pool
        self.conv1 = GCNConv(nfeat, nfeat)
        self.conv2 = GCNConv(nfeat, nfeat)
        self.attention = SelfAttention(nfeat*2)
        self.classifier = nn.Sequential(
            Linear(nfeat , nfeat),
            ELU(True),
            Linear(nfeat, nclass),
            ELU(True),
        )
    
    def forward(self, pooled_output, sent_labels): # [batch,6,768]
        datas = []
        for i in range(len(pooled_output)):
            x = pooled_output[i] # [6,768]

            edge_index = torch.arange(sent_labels[i].sum().item())
            edge_index = torch.cat([edge_index.unsqueeze(0).repeat(1,sent_labels[i].sum().item()),
                                    edge_index.unsqueeze(1).repeat(1,sent_labels[i].sum().item()).view(1,-1)],dim=0) # [2,36]
            edge_index1 = torch.cat([edge_index[1].unsqueeze(0),edge_index[0].unsqueeze(0)],dim=0)
            edge_index = torch.cat([edge_index,edge_index1],dim=1)
            edge_index = edge_index.to(x.device)
            data = Data(x=x, edge_index=edge_index)
            data.validate(raise_on_error=True)
            datas.append(data)
        datas = Batch.from_data_list(datas)
        x, edge_index = datas.x, datas.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.normalize(x,dim=-1)
        
        if self.pool == "att":
            x = x.view(-1,1+self.evi_max_num,x.shape[-1]) # [batch,6,768]
            feature_batch, claim_batch = x[:,1:,:], x[:,0,:] # [batch,5,768] # [batch,768]
            graph_rep = self.attention(feature_batch, claim_batch, sent_labels[:,1:]) # [batch,768]
        else:
            x = x.view(-1,self.evi_max_num,x.shape[-1]) # [batch,6,768]
            graph_rep = x.mean(dim=1) # [batch,768]

        outputs = self.classifier(graph_rep)
        return outputs

class CICR_graph(nn.Module):
    def __init__(self, nfeat, nclass, evi_max_num):
        super(CICR_graph, self).__init__()
        self.evi_max_num = evi_max_num
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.evidence_model = ONE_ATTENTION(nfeat, nclass, evi_max_num, "mean")
        self.fusion_model = ONE_ATTENTION(nfeat, nclass, evi_max_num, "att")
        self.classifier_claim = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        # constant_fusion = torch.nn.Parameter(torch.zeros((nfeat)))
        self.constant = nn.Parameter(torch.tensor(0.0))
        self.D_u = torch.randn((nclass,nfeat))
        self.linear1 = Linear(nfeat,64)
        self.linear2 = Linear(nfeat,64)
        self.linear3 = Linear(nfeat,nfeat)
    
    def claim_intervention(self, claims): # [batch,768]
        D_u = self.D_u.to(claims.device) # [3,768]
        L = self.linear1(claims) # [batch,64]
        K = self.linear2(D_u) # [3,64]
        w = torch.matmul(L,K.transpose(0,1)) # [batch,3]
        w = F.softmax(w,dim=-1)
        E_D_u = torch.matmul(w,D_u) # [batch,768]
        claims = self.linear3(claims + E_D_u)
        return claims

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels, indexs = data
        input_ids = input_ids.view(-1,input_ids.shape[-1])
        input_mask = input_mask.view(-1,input_ids.shape[-1])
        segment_ids = segment_ids.view(-1,input_ids.shape[-1])
        _, pooled_output = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,)
        pooled_output = pooled_output.view(-1,1+self.evi_max_num,pooled_output.shape[-1]) # [batch,1+5,768]
        claims = pooled_output[:,0,:] # [batch,768]

        claim_evidences = pooled_output # [batch,6,768]

        evidences = pooled_output[:,1:,:] # [batch,5,768]
      
        claims = claims.detach()
        claims = self.claim_intervention(claims)
        evidences = evidences.detach()

        res_claim = self.classifier_claim(claims) # [batch,3]
        res_evidence = self.evidence_model(evidences,sent_labels[:,1:]) # [batch,3]
        res_fusion = self.fusion_model(claim_evidences,sent_labels) # [batch,3]
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_evidence + res_fusion))

        counterfactual_final = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_evidence) \
             + self.constant * torch.ones_like(res_fusion)))
        TIE = res_final - counterfactual_final

        return res_claim, res_evidence, res_final, counterfactual_final, TIE


class CICR(nn.Module):
    def __init__(self, nfeat, nclass):
        super(CICR, self).__init__()
        self.bert = BertModel.from_pretrained("pretrained_models/BERT-Pair")
        self.classifier_claim = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        self.classifier_evidence = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        # constant_evidence = torch.nn.Parameter(torch.zeros((nfeat)))
        self.classifier_fusion = nn.Sequential(
            Linear(nfeat, nfeat),
            ReLU(True),
            Linear(nfeat, nclass),
            ReLU(True),
        )
        # constant_fusion = torch.nn.Parameter(torch.zeros((nfeat)))
        self.constant = nn.Parameter(torch.tensor(0.0))
        self.D_u = torch.randn((nclass,nfeat))
        self.linear1 = Linear(nfeat,64)
        self.linear2 = Linear(nfeat,64)
        self.linear3 = Linear(nfeat,nfeat)

    
    def claim_intervention(self, claims): # [batch,768]
        D_u = self.D_u.to(claims.device) # [3,768]
        L = self.linear1(claims) # [batch,64]
        K = self.linear2(D_u) # [3,64]
        w = torch.matmul(L,K.transpose(0,1)) # [batch,3]
        w = F.softmax(w,dim=-1)
        E_D_u = torch.matmul(w,D_u) # [batch,768]
        claims = self.linear3(claims + E_D_u)
        return claims
      

    def forward(self, data):
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,0,:] # [batch,128]
        input_mask = input_mask[:,0,:] # [batch,128]
        segment_ids = segment_ids[:,0,:] # [batch,128]
        _, claims = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        
        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,1,:] # [batch,128]
        input_mask = input_mask[:,1,:] # [batch,128]
        segment_ids = segment_ids[:,1,:] # [batch,128]
        _, claim_evidences = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]

        input_ids, input_mask, segment_ids, labels = data
        input_ids = input_ids[:,2,:] # [batch,128]
        input_mask = input_mask[:,2,:] # [batch,128]
        segment_ids = segment_ids[:,2,:] # [batch,128]
        _, evidences = self.bert(input_ids, token_type_ids=segment_ids, \
                                     attention_mask=input_mask, output_all_encoded_layers=False,) # [batch,768]
        claims = claims.detach()
        claims = self.claim_intervention(claims)
        evidences = evidences.detach()

        res_claim = self.classifier_claim(claims) # [batch,3]
        res_evidence = self.classifier_evidence(evidences) # [batch,3]
        res_fusion = self.classifier_fusion(claim_evidences) # [batch,3]
        res_final = torch.log(1e-8 + torch.sigmoid(res_claim + res_evidence + res_fusion))

        counterfactual_final = torch.log(1e-8 + torch.sigmoid(res_claim.detach() + self.constant * torch.ones_like(res_evidence) \
             + self.constant * torch.ones_like(res_fusion)))
        TIE = res_final - counterfactual_final

        return res_claim, res_evidence, res_final, counterfactual_final, TIE
