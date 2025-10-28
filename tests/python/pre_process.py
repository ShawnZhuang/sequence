import numpy as np



def pre_process(data:np.ndarray) -> (np.ndarray, np.ndarray): 
    """预处理函数：计算增长率""" 
    delta_ratio = (data[ 1:,:] - data[ :-1,:]) / data[:-1,:]
    return  data[0,:], delta_ratio



def softmax(x, axis=-1):
    """Softmax function along specified axis"""
    x = x - np.max(x, axis=axis, keepdims=True)  # 数值稳定性
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

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
    d_k = Q.shape[-1]  # 键的维度
    scores = np.dot(Q, K.T) / np.sqrt(d_k)  # (n_q, n_k)
    tri_mask = np.tril(np.ones_like(scores), k=0)    
    print("mask: {}".format(tri_mask.round(2)))
    scores= np.where(tri_mask, scores,-1e9)
    weights = softmax(scores, axis=-1)      # (n_q, n_k)
    print("scores after mask: {}".format(weights.round(2)))          # (n_q, d_v)
    print(weights.shape, V.shape )
    output=np.matmul(weights, V)    
    return output, weights


# 示例数据
np.random.seed(42)
num_seqs = 8
num_hidden = 4
data = np.random.rand(num_seqs, num_hidden) * 10 + 1  # 避免接近 0
 
print("原始 data:")
print(data)
initial_values, growth_rates = pre_process(data)

print("\n结果 (第0列=原始值，第1列起=增长率):")
print(data.shape)
print(initial_values.shape)
print(growth_rates.shape)

attended_output, weight=scaled_dot_product_attention(growth_rates,growth_rates,growth_rates)

print("\n注意力机制输出: {}".format(attended_output.shape))
print("最后一个的",attended_output[-1,:])
 

entropy = -np.sum(weight * np.log(weight + 1e-9), axis=-1)  # 添加小值以避免 log(0)