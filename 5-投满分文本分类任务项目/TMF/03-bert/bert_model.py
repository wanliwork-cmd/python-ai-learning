# Bert预训练模型 + 我们自定义的网络层 -> 分类的目的
import torch
import torch.nn as nn
from config import Config
from transformers import BertModel
from transformers import BertConfig

config = Config()

class BertClassifierModel(nn.Module):
    def __init__(self):
        # 1- 初始化父类
        super().__init__()

        # 2- 搭建网络结构
        # 2.1- 定义预训练模型
        self.bert_model = BertModel.from_pretrained(config.bert_path)
        for param in self.bert_model.parameters():
            param.requires_grad_(False)

        # 2.2- 定义我们自己的网络结构：层数自己决定
        # 获得预训练模型的隐藏状态向量维度
        bert_hidden_size = BertConfig.from_pretrained(config.bert_path).hidden_size
        self.linear = nn.Linear(in_features=bert_hidden_size, out_features=config.classname_cnt)

    def forward(self,input_ids,attention_mask,token_type_ids):
        # 1- 特征数据首先经过预训练模型
        """
            with torch.no_grad()
            作用：禁用模型的参数更新，也就是不允许进行反向传播
            使用：
                不加：训练后模型的效果会比较好；但是比较耗时，耗资源。实际工作，推荐不写
                加：训练速度快；但是训练后模型的效果可能一般般
        """
        with torch.no_grad():
            bert_output = self.bert_model(input_ids,attention_mask,token_type_ids)

        # 2- 预训练模型的输出，再给到我们自定义的网络结构处理
        """
            last_hidden_state[:,0]和pooler_output，实际是类似的东西，都表示[CLS]的隐藏状态。
            区别：需要对last_hidden_state[:,0]经过nn.Linear和激活函数处理后，才能得到pooler_output
            对应源代码位置：BertModel文件的697行
            【推荐】：使用last_hidden_state[:,0]
        """
        return self.linear(bert_output.last_hidden_state[:, 0])


