# 示例数据
import torch
from sequence.block.attention import scaled_dot_product_attention
torch.manual_seed(42)
num_seqs = 8
num_hidden = 4
data = torch.rand((num_seqs, num_hidden)) * 10 + 1  # 避免接近 0    
print("原始 data:")
print(data) 
initial_values = data[0, :]
growth_rates = (data[1:, :] - data[:-1, :]) / data[:-1, :]
print("\n结果 (第0列=原始值，第1列起=增长率):")
print(data.size())
print(initial_values.size())
print(growth_rates.size())
attended_output, weight=scaled_dot_product_attention(growth_rates,growth_rates,growth_rates)
print("\n注意力机制输出: {}".format(attended_output.size()))
print("最后一个的",attended_output[-1,:])