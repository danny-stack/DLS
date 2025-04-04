import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Pre-Layer Normalization (Pre-LN)
# -------------------------------

def rms_norm(x, eps=1e-6):
    """
    实现RMSNorm (Root Mean Square Normalization)
    在LLaMA等模型中使用的预归一化变体
    
    参数:
    x: 输入张量 [batch_size, dim]
    eps: 防止除零的小常数
    
    返回:
    normalized_x: 归一化后的张量
    """
    # 计算均方根
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    # 归一化
    return x / rms

def layer_norm(x, eps=1e-6):
    """
    实现Layer Normalization
    
    参数:
    x: 输入张量 [batch_size, dim]
    eps: 防止除零的小常数
    
    返回:
    normalized_x: 归一化后的张量
    """
    # 计算均值和方差
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    # 归一化
    return (x - mean) / np.sqrt(var + eps)

def transformer_block_with_pre_ln(x, W_q, W_k, W_v, W_o, W_ffn1, W_ffn2, use_rms=True):
    """
    实现带有Pre-LN的Transformer Block
    
    参数:
    x: 输入张量
    W_q, W_k, W_v, W_o: 自注意力的权重矩阵
    W_ffn1, W_ffn2: 前馈网络的权重矩阵
    use_rms: 是否使用RMSNorm (True) 或 LayerNorm (False)
    
    返回:
    output: Transformer Block的输出
    """
    # 选择归一化函数
    norm_fn = rms_norm if use_rms else layer_norm
    
    # 1. 自注意力层 (Self-Attention) 使用Pre-LN
    x_norm = norm_fn(x)
    
    # 计算Q, K, V
    q = x_norm @ W_q
    k = x_norm @ W_k
    v = x_norm @ W_v
    
    # 计算注意力
    scores = q @ k.T
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    attention_output = attention_weights @ v
    
    # 投影输出
    attention_output = attention_output @ W_o
    
    # 残差连接
    x = x + attention_output
    
    # 2. 前馈网络 (FFN) 使用Pre-LN
    x_norm = norm_fn(x)
    
    # 两层前馈网络
    ffn_output = x_norm @ W_ffn1
    # SwiGLU激活函数 (LLaMA中使用)
    ffn_output = swiglu(ffn_output)
    ffn_output = ffn_output @ W_ffn2
    
    # 残差连接
    output = x + ffn_output
    
    return output

# -------------------------------
# 2. SwiGLU激活函数
# -------------------------------

def swiglu(x):
    """
    实现SwiGLU激活函数
    PaLM和LLaMA中使用的激活函数变体
    
    参数:
    x: 输入张量 [batch_size, 2*dim]
    
    返回:
    output: 激活后的张量 [batch_size, dim]
    """
    half_dim = x.shape[-1] // 2
    x1, x2 = x[:, :half_dim], x[:, half_dim:]
    
    # SwiGLU: swish(x1) * x2
    # 其中swish(x) = x * sigmoid(x)
    sigmoid_x1 = 1 / (1 + np.exp(-x1))
    swish_x1 = x1 * sigmoid_x1
    
    return swish_x1 * x2

# -------------------------------
# 3. 实现KV-Cache（推理阶段优化）
# -------------------------------

class KVCache:
    """
    实现KV-Cache，用于加速自回归生成
    
    这是现代大模型推理优化的核心技术之一
    """
    def __init__(self, max_seq_len, dim):
        """
        初始化KV-Cache
        
        参数:
        max_seq_len: 最大序列长度
        dim: 缓存的维度
        """
        self.max_seq_len = max_seq_len
        self.dim = dim
        # 初始化K和V的缓存
        self.k_cache = np.zeros((max_seq_len, dim))
        self.v_cache = np.zeros((max_seq_len, dim))
        # 当前位置
        self.current_len = 0
    
    def update(self, k, v):
        """
        更新缓存
        
        参数:
        k: 当前位置的key向量
        v: 当前位置的value向量
        """
        if self.current_len < self.max_seq_len:
            self.k_cache[self.current_len] = k
            self.v_cache[self.current_len] = v
            self.current_len += 1
        else:
            raise ValueError("KV Cache已满")
    
    def get_cached_kv(self):
        """获取当前缓存的所有k和v"""
        return self.k_cache[:self.current_len], self.v_cache[:self.current_len]

def self_attention_with_kv_cache(q, kv_cache, W_v, new_k=None, new_v=None):
    """
    带KV-Cache的自注意力计算
    
    参数:
    q: 当前的query向量
    kv_cache: KVCache对象
    W_v: Value的权重矩阵
    new_k, new_v: 当前token的key和value向量（如果需要更新缓存）
    
    返回:
    output: 自注意力输出
    """
    # 如果提供了新的k和v，则更新缓存
    if new_k is not None and new_v is not None:
        kv_cache.update(new_k, new_v)
    
    # 获取缓存的k和v
    k_cached, v_cached = kv_cache.get_cached_kv()
    
    # 计算注意力分数
    scores = q @ k_cached.T  # [1, current_len]
    
    # 生成掩码（仅使用已知的token）
    mask = np.triu(np.ones((kv_cache.current_len, kv_cache.current_len)), k=1).astype(bool)
    scores_masked = scores.copy()
    scores_masked[:, mask[-1]] = -1e9  # 掩盖未来token
    
    # 归一化
    weights = np.exp(scores_masked) / np.sum(np.exp(scores_masked))
    
    # 计算输出
    output = weights @ v_cached
    
    return output

# -------------------------------
# 4. FlashAttention原理演示（计算效率优化）
# -------------------------------

def naive_attention(Q, K, V):
    """
    传统的自注意力计算（用于对比）
    
    参数:
    Q, K, V: 查询、键、值矩阵
    
    返回:
    output: 注意力输出
    """
    # 计算注意力分数
    scores = Q @ K.T  # [seq_len, seq_len]
    
    # Softmax归一化
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # 计算输出
    output = weights @ V
    
    return output

def block_sparse_attention_demo(Q, K, V, block_size=2):
    """
    基于块的稀疏注意力演示
    FlashAttention/块稀疏注意力的简化版本
    
    参数:
    Q, K, V: 查询、键、值矩阵
    block_size: 块大小
    
    返回:
    output: 注意力输出
    """
    seq_len = Q.shape[0]
    output = np.zeros_like(Q)
    
    # 按块计算
    for i in range(0, seq_len, block_size):
        Q_block = Q[i:i+block_size]
        
        # 存储当前块的归一化系数
        normalizer = np.zeros((min(block_size, seq_len-i), 1))
        
        for j in range(0, seq_len, block_size):
            K_block = K[j:j+block_size]
            V_block = V[j:j+block_size]
            
            # 计算当前块的注意力分数
            scores_block = Q_block @ K_block.T  # [block, block]
            
            # 指数化但不归一化
            exp_scores = np.exp(scores_block)
            
            # 更新归一化系数
            normalizer += np.sum(exp_scores, axis=-1, keepdims=True)
            
            # 累积输出
            output[i:i+block_size] += exp_scores @ V_block
        
        # 归一化
        output[i:i+block_size] /= normalizer
    
    return output

# -------------------------------
# 5. Grouped-Query Attention (GQA)
# -------------------------------

def grouped_query_attention(x, W_q, W_k, W_v, num_heads=4, num_kv_heads=2):
    """
    实现分组查询注意力 (Grouped-Query Attention)
    用于减少KV缓存的大小
    
    参数:
    x: 输入张量 [seq_len, d_model]
    W_q, W_k, W_v: 权重矩阵
    num_heads: Q头的数量
    num_kv_heads: K和V头的数量（小于等于num_heads）
    
    返回:
    output: 注意力输出
    """
    seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    
    # 确保num_heads可以被num_kv_heads整除
    assert num_heads % num_kv_heads == 0, "Q头数必须是KV头数的整数倍"
    
    # 每个KV头对应多少个Q头
    num_q_per_kv = num_heads // num_kv_heads
    
    # 计算Q, K, V
    q = x @ W_q  # [seq_len, d_model]
    k = x @ W_k  # [seq_len, (d_model//num_heads)*num_kv_heads]
    v = x @ W_v  # [seq_len, (d_model//num_heads)*num_kv_heads]
    
    # 重塑为多头形式
    q = q.reshape(seq_len, num_heads, head_dim)  # [seq_len, num_heads, head_dim]
    k = k.reshape(seq_len, num_kv_heads, head_dim)  # [seq_len, num_kv_heads, head_dim]
    v = v.reshape(seq_len, num_kv_heads, head_dim)  # [seq_len, num_kv_heads, head_dim]
    
    # 初始化输出
    output = np.zeros((seq_len, num_heads, head_dim))
    
    # 为每个Q头计算注意力
    for q_head in range(num_heads):
        # 确定使用哪个KV头
        kv_head = q_head // num_q_per_kv
        
        # 当前头的Q
        q_h = q[:, q_head, :]  # [seq_len, head_dim]
        
        # 对应的K和V
        k_h = k[:, kv_head, :]  # [seq_len, head_dim]
        v_h = v[:, kv_head, :]  # [seq_len, head_dim]
        
        # 计算注意力分数
        scores = q_h @ k_h.T  # [seq_len, seq_len]
        
        # Softmax归一化
        weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # 计算输出
        output[:, q_head, :] = weights @ v_h
    
    # 重塑回原始形状并返回
    return output.reshape(seq_len, d_model)

# -------------------------------
# 6. 演示使用对比
# -------------------------------

def compare_mechanisms():
    """演示并对比不同机制的行为差异"""
    # 创建示例数据
    seq_len = 4
    d_model = 16
    
    # 随机特征
    np.random.seed(42)
    x = np.random.normal(0, 1, (seq_len, d_model))
    
    # 归一化对比
    x_layer_norm = layer_norm(x)
    x_rms_norm = rms_norm(x)
    
    # 激活函数对比
    x_for_act = np.random.normal(0, 1, (seq_len, d_model*2))
    x_swiglu = swiglu(x_for_act)
    
    # 打印结果
    print("原始输入:")
    print(x[0, :5])  # 只打印第一行的前几个值
    
    print("\nLayerNorm后:")
    print(x_layer_norm[0, :5])
    
    print("\nRMSNorm后:")
    print(x_rms_norm[0, :5])
    
    print("\nSwiGLU输出:")
    print(x_swiglu[0, :5])
    
    # 绘制归一化效果对比
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.title("原始特征")
    plt.imshow(x, cmap="viridis")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("LayerNorm")
    plt.imshow(x_layer_norm, cmap="viridis")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title("RMSNorm")
    plt.imshow(x_rms_norm, cmap="viridis")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    # 可视化SwiGLU效果
    x_range = np.linspace(-3, 3, 100)
    x_in = np.column_stack((x_range, x_range))
    
    sigmoid = 1 / (1 + np.exp(-x_range))
    swish = x_range * sigmoid
    glu = x_range * sigmoid * x_range  # 简化的SwiGLU示例
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, sigmoid, label='Sigmoid')
    plt.plot(x_range, swish, label='Swish')
    plt.plot(x_range, glu, label='SwiGLU-like')
    plt.grid(True)
    plt.legend()
    plt.title('SwiGLU及相关激活函数对比')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.show()
    
    # KV-Cache演示
    print("\nKV-Cache 演示:")
    seq_len = 8
    d_model = 4
    
    # 创建一个KV缓存
    kv_cache = KVCache(max_seq_len=seq_len, dim=d_model)
    
    # 随机生成一些示例数据
    np.random.seed(42)
    x_seq = np.random.normal(0, 1, (seq_len, d_model))
    W_q = np.random.normal(0, 0.1, (d_model, d_model))
    W_k = np.random.normal(0, 0.1, (d_model, d_model))
    W_v = np.random.normal(0, 0.1, (d_model, d_model))
    
    # 模拟自回归生成过程
    outputs = []
    for i in range(4):  # 只演示前4个token
        # 当前token的q, k, v
        q_i = x_seq[i:i+1] @ W_q
        k_i = x_seq[i:i+1] @ W_k
        v_i = x_seq[i:i+1] @ W_v
        
        # 使用KV缓存计算注意力
        output = self_attention_with_kv_cache(q_i, kv_cache, W_v, new_k=k_i[0], new_v=v_i[0])
        outputs.append(output)
        
        print(f"处理Token {i+1}, 当前缓存大小: {kv_cache.current_len}")
    
    print(f"最终缓存大小: {kv_cache.current_len}")
    
    # 对比GQA和普通多头注意力
    print("\nGQA vs 标准多头注意力:")
    seq_len = 4
    d_model = 16
    num_heads = 4
    num_kv_heads = 2
    
    # 随机输入
    x = np.random.normal(0, 1, (seq_len, d_model))
    
    # 权重矩阵
    W_q_std = np.random.normal(0, 0.1, (d_model, d_model))
    W_k_std = np.random.normal(0, 0.1, (d_model, d_model))
    W_v_std = np.random.normal(0, 0.1, (d_model, d_model))
    
    W_q_gqa = np.random.normal(0, 0.1, (d_model, d_model))
    W_k_gqa = np.random.normal(0, 0.1, (d_model, d_model * num_kv_heads // num_heads))
    W_v_gqa = np.random.normal(0, 0.1, (d_model, d_model * num_kv_heads // num_heads))
    
    # 计算内存使用差异
    std_memory = d_model * (2 * num_heads)  # K和V各需要一个头
    gqa_memory = d_model * (2 * num_kv_heads)  # K和V只需要较少的头
    
    print(f"标准多头注意力KV内存: {std_memory} 个参数")
    print(f"GQA内存: {gqa_memory} 个参数")
    print(f"内存节省: {(1 - gqa_memory/std_memory)*100:.1f}%")

# -------------------------------
# 7. 增加Sliding Window Attention机制
# -------------------------------

def sliding_window_attention(Q, K, V, window_size=2):
    """
    实现滑动窗口注意力机制
    仅考虑每个位置前后window_size范围内的token
    
    参数:
    Q, K, V: 注意力矩阵
    window_size: 窗口大小 (单侧)
    
    返回:
    output: 注意力输出
    """
    seq_len = Q.shape[0]
    output = np.zeros_like(Q)
    
    # 为每个位置计算滑动窗口注意力
    for i in range(seq_len):
        # 计算窗口的开始和结束索引
        start_idx = max(0, i - window_size)
        end_idx = min(seq_len, i + window_size + 1)
        
        # 当前位置的query
        q_i = Q[i:i+1]  # [1, dim]
        
        # 窗口内的key和value
        k_window = K[start_idx:end_idx]  # [window_size*2, dim]
        v_window = V[start_idx:end_idx]  # [window_size*2, dim]
        
        # 计算注意力分数
        scores = q_i @ k_window.T  # [1, window_size*2]
        
        # Softmax归一化
        weights = np.exp(scores) / np.sum(np.exp(scores))
        
        # 计算输出
        output[i] = weights @ v_window
    
    return output

# -------------------------------
# 8. MoE (Mixture of Experts) 简化实现
# -------------------------------

class SimpleMoE:
    """
    简化版的Mixture of Experts实现
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4, top_k=2):
        """
        初始化MoE层
        
        参数:
        input_dim: 输入维度
        hidden_dim: 专家隐藏层维度
        output_dim: 输出维度
        num_experts: 专家数量
        top_k: 每次激活的专家数量
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 初始化门控网络
        self.gate = np.random.normal(0, 0.1, (input_dim, num_experts))
        
        # 初始化专家网络
        self.experts = []
        for _ in range(num_experts):
            # 每个专家有两层权重
            w1 = np.random.normal(0, 0.1, (input_dim, hidden_dim))
            w2 = np.random.normal(0, 0.1, (hidden_dim, output_dim))
            self.experts.append((w1, w2))
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        x: 输入张量 [batch_size, input_dim]
        
        返回:
        output: MoE层输出
        """
        batch_size = x.shape[0]
        
        # 计算门控得分
        gate_scores = x @ self.gate  # [batch_size, num_experts]
        
        # 找到top_k专家
        expert_indices = np.argsort(gate_scores, axis=1)[:, -self.top_k:]  # [batch_size, top_k]
        
        # 计算softmax权重
        expert_weights = np.zeros((batch_size, self.num_experts))
        for i in range(batch_size):
            for j in expert_indices[i]:
                expert_weights[i, j] = np.exp(gate_scores[i, j])
            # 归一化
            expert_weights[i] /= np.sum(expert_weights[i])
        
        # 初始化输出
        output = np.zeros((batch_size, self.output_dim))
        
        # 计算每个专家的输出并加权求和
        for i in range(batch_size):
            for j in range(self.num_experts):
                if expert_weights[i, j] > 0:
                    # 获取当前专家的权重
                    w1, w2 = self.experts[j]
                    
                    # 计算当前专家的输出
                    hidden = np.maximum(0, x[i] @ w1)  # ReLU激活
                    expert_output = hidden @ w2
                    
                    # 加权求和
                    output[i] += expert_weights[i, j] * expert_output
        
        return output

def demo_moe():
    """演示MoE的工作原理"""
    # 创建一个简单的MoE层
    input_dim = 8
    hidden_dim = 16
    output_dim = 4
    num_experts = 4
    top_k = 2
    
    moe = SimpleMoE(input_dim, hidden_dim, output_dim, num_experts, top_k)
    
    # 随机输入
    np.random.seed(42)
    x = np.random.normal(0, 1, (3, input_dim))
    
    # 前向传播
    output = moe.forward(x)
    
    print("\nMoE演示:")
    print(f"输入shape: {x.shape}")
    print(f"输出shape: {output.shape}")
    print(f"专家数量: {num_experts}, 激活的专家: {top_k}")
    
    # 计算每个输入激活了哪些专家
    gate_scores = x @ moe.gate
    expert_indices = np.argsort(gate_scores, axis=1)[:, -top_k:]
    
    for i in range(len(x)):
        print(f"样本 {i+1} 激活的专家: {expert_indices[i]}")

# 如果直接运行此脚本，则执行演示
if __name__ == "__main__":
    compare_mechanisms()
    demo_moe()
    
    # 测试滑动窗口注意力
    seq_len = 8
    d_model = 4
    np.random.seed(42)
    
    Q = np.random.normal(0, 1, (seq_len, d_model))
    K = np.random.normal(0, 1, (seq_len, d_model))
    V = np.random.normal(0, 1, (seq_len, d_model))
    
    # 全局注意力vs滑动窗口
    full_attention = naive_attention(Q, K, V)
    window_attention = sliding_window_attention(Q, K, V, window_size=2)
    
    print("\n滑动窗口注意力vs全局注意力:")
    print(f"全局注意力输出 (前3行):\n{full_attention[:3]}")
    print(f"滑动窗口输出 (前3行):\n{window_attention[:3]}")
