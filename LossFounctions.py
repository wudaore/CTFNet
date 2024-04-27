def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean'):
    # 归一化
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    # 计算查询向量与正样本键向量的内积，得到原始的对比分数
    logits = query @ transpose(positive_key)
    # 构造标签，即每个查询向量对应的标签为其在查询向量集合中的索引
    labels = torch.arange(len(query), device=query.device)
    # INFONce是通过交叉熵实现的
    loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)
    return loss



# 监督对比损失
def SupConLoss(features, labels=None, mask=None):
    device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))

    temperature = 0.07
    contrast_mode = 'all'
    base_temperature = 0.07

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    mask = mask.repeat(anchor_count, contrast_count)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    
    weight = 1 - torch.exp(log_prob)
    weight = weight * weight
    mean_log_prob_pos = (weight * mask * log_prob).sum(1) / mask.sum(1)
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss