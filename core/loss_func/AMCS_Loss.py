import math
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.nn import Parameter
from core.utils import accuracy, PRF
from .finetuning_model import FinetuningModel
import numpy as np

# Computing S_B and S_W
def compute_J(encode_images, test_y, signl):
    """
    compute the sb、 sw  and  J .
    :param encode_images: the images after encoding
    :param test_y: the label.
    :return:trace(sw) 、 trace(sb) and J
    """
    if signl == "t":
        test_y = test_y
    else:
        test_y = test_y.view(-1)
    encode_images = encode_images.cpu().detach().numpy()
    num_samples1 = np.sum(test_y.cpu().numpy() == 0)
    num_samples2 = np.sum(test_y.cpu().numpy() == 1)

    test_y = torch.zeros((len(test_y), 2)).to(test_y.device).scatter_(1, test_y.unsqueeze(1), 1)
    test_y = test_y.cpu().numpy()
    _ , n_feature = encode_images.shape
    row,column=test_y.shape

    test = []
    test.append(np.zeros(shape=(num_samples1,n_feature)))
    test.append(np.zeros(shape=(num_samples2,n_feature)))

    P=np.zeros(shape=(column,))
    m=np.zeros(shape=(column,n_feature))
    index=np.argmax(test_y,axis=1)
    sw=0

    for i in range(column):
        test[i] = encode_images[index==i]
        P[i]=len(test[i])/row
        m[i]=np.mean(test[i],axis=0)
        sw=sw+P[i]*np.cov(test[i],rowvar=0)

    for i in range(column):
        m[i]=P[i]*m[i]
    m0=np.sum(m,axis=0)

    sb=0
    for i in range(column):
        t1=(m[i]-m0).reshape(1,n_feature)
        t2 =(m[i]-m0).reshape(n_feature,1)
        sb=sb+P[i]*np.dot(t2,t1)
    J = np.trace(sb)/(np.trace(sw)+0.000001)
    return np.trace(sb), np.trace(sw), J


class Margin_Classifier(nn.Module):
    # Implement of negative margin cosine distance

    def __init__(self, in_features, out_features, scale_factor= 40, margin=-0.05):
        super(Margin_Classifier, self).__init__()
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        if label is None:
            return cosine * self.scale_factor

        phi = cosine - self.margin
        output = torch.where(
            self.one_hot(label, cosine.shape[1]).byte(), phi, cosine)
        output *= self.scale_factor
        return output

    def one_hot(self, y, num_class):
        return (
            torch.zeros((len(y), num_class)).to(y.device).scatter_(1, y.unsqueeze(1), 1)
        )


class AMCS_Loss(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param, **kwargs):
        super(AMCS_Loss, self).__init__(**kwargs)

        self.feat_dim = feat_dim
        self.num_class = num_class
        self.inner_param = inner_param

        self.loss_func = nn.CrossEntropyLoss()
        self.classifier = Margin_Classifier(self.feat_dim, self.num_class)

    def set_forward(self, batch):
        log_feature, global_target = batch
        log_feature = log_feature.to(self.device)
        with torch.no_grad():
            feat = self.emb_func(log_feature.view(-1, 300, 300))
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        episode_size = support_feat.size(0)
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(
                support_feat[i], support_target[i], query_feat[i]
            )
            output_list.append(output)
        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))
        res = PRF(output, query_target.reshape(-1))
        return output, acc, res


    def set_forward_loss(self, batch):
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)

        feat = self.emb_func(image.view(-1,300,300))
        output = self.classifier(feat,target)
        loss = self.loss_func(output, target.reshape(-1))
        acc = accuracy(output, target.reshape(-1))
        return output, acc, loss

    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        classifier = Margin_Classifier(self.feat_dim, self.way_num)
        optimizer = self.sub_optimizer(classifier, self.inner_param["inner_optim"])

        classifier = classifier.to(self.device)

        classifier.train()
        support_size = support_feat.size(0)
        for epoch in range(self.inner_param["inner_train_iter"]):
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, self.inner_param["inner_batch_size"]):
                select_id = rand_id[
                    i : min(i + self.inner_param["inner_batch_size"], support_size)
                ]
                batch = support_feat[select_id]
                target = support_target[select_id]

                output = classifier(batch,target)
                loss = self.loss_func(output, target)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        output = classifier(query_feat)
        return output
