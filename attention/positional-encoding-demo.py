import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. 原始Transformer的正弦余弦位置编码
# -------------------------------

def sinusoidal_positional_encoding(positions, d_model):
    """
    实现Transformer原始论文中的正弦余弦位置编码
    
    参数:
    positions: 位置索引范围
    d_model: 模型维度
    
    返回:
    positional_encoding: [max_position, d_model]的位置编码矩阵
    """
    # 创建一个空矩阵来存储位置编码
    max_position = len(positions)
    positional_encoding = np.zeros((max_position, d_model))
    
    # 计算不同维度的频率
    for i in positions:
        for j in range(0, d_model, 2):
            # 每个维度使用不同频率的正弦/余弦函数
            freq = 1.0 / (10000 ** (j / d_model))
            
            # 偶数维度使用sin
            positional_encoding[i, j] = np.sin(i * freq)
            
            # 奇数维度使用cos（如果还在维度范围内）
            if j + 1 < d_model:
                positional_encoding[i, j + 1] = np.cos(i * freq)
    
    return positional_encoding

# 可视化正弦余弦位置编码
def plot_sinusoidal_encoding():
    max_position = 20
    d_model = 32
    
    positions = np.arange(max_position)
    pos_encoding = sinusoidal_positional_encoding(positions, d_model)
    
    plt.figure(figsize=(15, 8))
    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    plt.title("Sinusoidal Positional Encoding")
    plt.show()
    
    # 绘制几个位置的编码曲线
    plt.figure(figsize=(15, 5))
    for pos in [0, 5, 10, 15]:
        plt.plot(pos_encoding[pos, :], label=f"Position {pos}")
    plt.legend()
    plt.title("Position Encoding Curves for Different Positions")
    plt.xlabel("Dimension")
    plt.ylabel("Encoding Value")
    plt.grid(True)
    plt.show()

# -------------------------------
# 2. 可学习位置编码
# -------------------------------

def learned_positional_encoding(max_position, d_model, seed=42):
    """
    实现BERT中使用的可学习位置编码（这里用随机初始化模拟）
    
    参数:
    max_position: 最大位置数
    d_model: 模型维度
    seed: 随机种子
    
    返回:
    positional_encoding: [max_position, d_model]的位置编码矩阵
    """
    np.random.seed(seed)
    # 在实际模型中，这将是可训练的参数
    return np.random.normal(0, 0.2, (max_position, d_model))

# -------------------------------
# 3. 旋转位置编码 (RoPE)
# -------------------------------

def get_rope_matrix(positions, d_model, base=10000):
    """
    生成RoPE(Rotary Position Embedding)的旋转矩阵
    
    参数:
    positions: 位置索引
    d_model: 特征维度
    base: RoPE的频率基数
    
    返回:
    cos_matrix, sin_matrix: 余弦和正弦旋转矩阵
    """
    max_position = len(positions)
    assert d_model % 2 == 0, "特征维度必须是偶数"
    
    half_d_model = d_model // 2
    
    # 计算每个维度的频率
    freq = 1.0 / (base ** (np.arange(0, half_d_model) / half_d_model))  # [half_d_model]
    
    # 将频率应用于每个位置
    # 对于每个位置pos和维度i，我们计算pos·θᵢ
    t = np.outer(positions, freq)  # [max_position, half_d_model]
    
    # 计算cos和sin
    cos_matrix = np.cos(t)  # [max_position, half_d_model]
    sin_matrix = np.sin(t)  # [max_position, half_d_model]
    
    return cos_matrix, sin_matrix

def apply_rope(x, cos_matrix, sin_matrix):
    """
    应用旋转位置编码到输入张量
    
    参数:
    x: 输入张量 [batch_size, seq_len, d_model]
    cos_matrix: 余弦矩阵 [seq_len, d_model/2]
    sin_matrix: 正弦矩阵 [seq_len, d_model/2]
    
    返回:
    output: 应用RoPE后的张量 [batch_size, seq_len, d_model]
    """
    seq_len, d_model = x.shape
    half_d_model = d_model // 2
    
    # 将特征分成两半
    x_reshape = x.reshape(seq_len, 2, half_d_model)
    x_1 = x_reshape[:, 0, :]  # [seq_len, half_d_model]
    x_2 = x_reshape[:, 1, :]  # [seq_len, half_d_model]
    
    # 应用旋转
    output_1 = x_1 * cos_matrix - x_2 * sin_matrix
    output_2 = x_2 * cos_matrix + x_1 * sin_matrix
    
    # 重新组合
    output = np.zeros_like(x)
    output[:, ::2] = output_1
    output[:, 1::2] = output_2
    
    return output

# 在实际应用中，RoPE通常应用于Query和Key
def rope_attention_qk(query, key, cos_matrix, sin_matrix):
    """
    将RoPE应用于Query和Key，然后计算注意力
    
    参数:
    query: [seq_len, d_model]
    key: [seq_len, d_model]
    cos_matrix, sin_matrix: RoPE旋转矩阵
    
    返回:
    attention_scores: [seq_len, seq_len]
    """
    # 应用RoPE到Q和K
    query_rope = apply_rope(query, cos_matrix, sin_matrix)
    key_rope = apply_rope(key, cos_matrix, sin_matrix)
    
    # 计算注意力分数
    return query_rope @ key_rope.T

# -------------------------------
# 4. 在原始自注意力实现中添加位置编码
# -------------------------------

def self_attention_with_position(Feature_all, W_q, W_k, W_v, position_encoding, add_position=True):
    """
    带位置编码的自注意力计算
    
    参数:
    Feature_all: 输入特征 [seq_len, d_model]
    W_q, W_k, W_v: 权重矩阵
    position_encoding: 位置编码 [seq_len, d_model]
    add_position: 是否使用位置编码
    
    返回:
    output: 自注意力输出
    weights: 注意力权重
    """
    # 如果使用位置编码，将其添加到输入特征中
    if add_position:
        Feature_with_pos = Feature_all + position_encoding
    else:
        Feature_with_pos = Feature_all
    
    # 计算Q、K、V
    Query = Feature_with_pos @ W_q
    Key = Feature_with_pos @ W_k
    Value = Feature_with_pos @ W_v
    
    # 计算注意力分数
    scores = Query @ Key.T
    
    # 归一化
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # 计算输出
    output = weights @ Value
    
    return output, weights

# -------------------------------
# 5. 使用RoPE的自注意力实现
# -------------------------------

def self_attention_with_rope(Feature_all, W_q, W_k, W_v, positions, d_model):
    """
    使用RoPE的自注意力计算
    
    参数:
    Feature_all: 输入特征 [seq_len, d_model]
    W_q, W_k, W_v: 权重矩阵
    positions: 位置索引
    d_model: 模型维度
    
    返回:
    output: 自注意力输出
    weights: 注意力权重
    """
    # 计算Q、K、V
    Query = Feature_all @ W_q
    Key = Feature_all @ W_k
    Value = Feature_all @ W_v
    
    # 获取RoPE旋转矩阵
    cos_matrix, sin_matrix = get_rope_matrix(positions, Query.shape[1])
    
    # 应用RoPE到Q和K
    Query_rope = apply_rope(Query, cos_matrix, sin_matrix)
    Key_rope = apply_rope(Key, cos_matrix, sin_matrix)
    
    # 计算注意力分数
    scores = Query_rope @ Key_rope.T
    
    # 归一化
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # 计算输出
    output = weights @ Value
    
    return output, weights

# -------------------------------
# 6. 演示使用
# -------------------------------

# 示例使用
if __name__ == "__main__":
    # 使用原有的特征数据
    Feature_all = np.array([
        [1, 0, 1],  # 位置1
        [0, 1, 1],  # 位置2  
        [1, 1, 0],  # 位置3
        [0, 0, 1]   # 位置4
    ])
    
    # 获取输入和输出维度
    n_in = 3   # 输入维度
    n_out = 2  # 输出维度
    seq_len = 4  # 序列长度
    
    # 初始化权重矩阵
    np.random.seed(42)
    std = np.sqrt(2.0 / (n_in + n_out))
    W_q = np.random.normal(0, std, (n_in, n_out))
    W_k = np.random.normal(0, std, (n_in, n_out))
    W_v = np.random.normal(0, std, (n_in, n_out))
    
    # 1. 不使用位置编码
    output_no_pos, weights_no_pos = self_attention_with_position(
        Feature_all, W_q, W_k, W_v, 
        position_encoding=np.zeros((seq_len, n_in)), 
        add_position=False
    )
    
    # 2. 使用正弦余弦位置编码
    positions = np.arange(seq_len)
    sin_cos_encoding = sinusoidal_positional_encoding(positions, n_in)
    output_sincos, weights_sincos = self_attention_with_position(
        Feature_all, W_q, W_k, W_v, 
        position_encoding=sin_cos_encoding, 
        add_position=True
    )
    
    # 3. 使用可学习位置编码
    learned_encoding = learned_positional_encoding(seq_len, n_in)
    output_learned, weights_learned = self_attention_with_position(
        Feature_all, W_q, W_k, W_v, 
        position_encoding=learned_encoding, 
        add_position=True
    )
    
    # 4. 使用RoPE位置编码
    output_rope, weights_rope = self_attention_with_rope(
        Feature_all, W_q, W_k, W_v, 
        positions=positions, 
        d_model=n_out
    )
    
    # 比较不同方法的注意力权重
    print("无位置编码的注意力权重:")
    print(weights_no_pos)
    print("\n正弦余弦位置编码的注意力权重:")
    print(weights_sincos)
    print("\n可学习位置编码的注意力权重:")
    print(weights_learned)
    print("\nRoPE位置编码的注意力权重:")
    print(weights_rope)
    
    # 可视化正弦余弦位置编码
    plot_sinusoidal_encoding()
