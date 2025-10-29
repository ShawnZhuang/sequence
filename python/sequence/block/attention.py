import torch 


def scaled_dot_product_attention(Q, K, V):
    """
    计算缩放点积注意力
    参数:
        Q: (n_queries, d_k)      查询矩阵
        K: (n_keys,    d_k)      键矩阵
        V: (n_keys,    d_v)      值矩阵
    返回:
        output: (n_queries, d_v)  注意力输出
        weights: (n_queries, n_keys) 注意力权重
    """
    d_k = Q.size(-1)  # 键的维度
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # (n_q, n_k)
    tri_mask = torch.tril(torch.ones_like(scores), diagonal=0)    
    print("mask: {}".format(tri_mask))
    scores= torch.where(tri_mask==1, scores,torch.tensor(-1e9))
    weights = torch.softmax(scores, dim=-1)      # (n_q, n_k)
    print("scores after mask: {}".format(weights))          # (n_q, d_v)
    print(weights.size(), V.size() )
    output=torch.matmul(weights, V)    
    return output, weights

