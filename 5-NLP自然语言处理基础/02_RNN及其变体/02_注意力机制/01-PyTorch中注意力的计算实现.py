import torch
import torch.nn as nn

class MyAttn(nn.Module):
    def __init__(self,query_size,key_size,value_size,weighted_size,output_size):
        """
        :param query_size: 第1步中，进行拼接Q张量的最后一个维度，也就是8
        :param key_size:第1步中，进行拼接K张量的最后一个维度，也就是8
        :param value_size: 第4步中，进行专属信息包计算时，V张量的最后一个维度，也就是8
        :param weighted_size: 第2、3步中，相似性张量、权重张量的最后一个维度，也就是5
        :param output_size: 第6步中，对[1,1,16]形状调整后得到[1,1,8]中8对应位置
        """

        # 1- 初始化父类
        super().__init__()

        # 2- 设置属性值
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.weighted_size = weighted_size
        self.output_size = output_size

        # 3- 搭建神经网络
        # 3.1- 计算相似性的线性层。也就是原理图中第2步
        """
            in_features：输入到该线性层的向量维度，也就是Q和K拼接后的张量
        """
        self.attn_linear = nn.Linear(in_features=self.query_size+self.key_size, out_features=self.weighted_size)
        # self.attn_linear = nn.Linear(in_features=16,out_features=5)

        # 3.2- 调整张量的形状，达到GRU输入的要求。也就是原理图中的第6步
        self.attn_combine_linear = nn.Linear(in_features=self.query_size+self.key_size, out_features=self.output_size)
        # self.attn_combine_linear = nn.Linear(in_features=16, out_features=8)

    def forward(self,Q,K,V):
        """
        计算专属信息包
        :param Q: query
        :param K: key
        :param V: value
        :return:
        """
        # 1- QK拼接
        qk_concat = torch.concat((Q,K),dim=-1)
        print(f"第1步：{qk_concat.shape}")

        # 2- 计算Q和K的相似性：用的时注意力计算的第1个公式 注意力=softmax(linear([Q,K])) @ V
        attn_score = self.attn_linear(qk_concat)
        print(f"第2步：{attn_score.shape}")

        # 3- 通过Softmax将相似性转成权重
        attn_weight = torch.softmax(attn_score,dim=-1)
        print(f"第3步：{attn_weight.shape}")

        # 4- 权重矩阵和V进行计算，得到专属信息包
        C = torch.bmm(attn_weight, V)
        print(f"第4步：{C.shape}")

        # 5- 拼接上一个时间步的预测结果词向量 和 专属信息包
        qc_concat = torch.concat((Q,C),dim=-1)
        print(f"第5步：{qc_concat.shape}")

        # 6- 经过线性层对张量进行进行变换：[1,1,16]形状调整后得到[1,1,8]
        output = self.attn_combine_linear(qc_concat)
        print(f"第6步：{output.shape}")

        return output,attn_weight

if __name__ == '__main__':
    # 因为只讲PyTorch中注意力是如何计算得到。因此拼接后的value、Q、V我们直接手动初始化

    # 1- 定义基础变量
    batch_size = 1      # 每个批次中句子的条数
    seq_len = 5         # 每个句子中词的个数
    hidden_size = 8     # 我们人为设置词向量维度和隐藏向量维度相同
    num_layers = 1      # 隐藏层层数

    # 2- 初始化Q、K、V
    Q = torch.randn(size=(batch_size,num_layers,hidden_size))
    K = torch.randn(size=(batch_size,num_layers,hidden_size))
    V = torch.randn(size=(batch_size,seq_len,hidden_size))      # 多个词的隐藏状态拼接后的

    # 3- 计算注意力
    query_size = hidden_size
    key_size = hidden_size
    value_size = hidden_size
    weighted_size = seq_len
    output_size = hidden_size

    attn_model = MyAttn(query_size,key_size,value_size,weighted_size,output_size)
    output,attn_weight = attn_model(Q,K,V)

    print(f"权重值{attn_weight}")
    print(f"权重值和{attn_weight.sum()}")

