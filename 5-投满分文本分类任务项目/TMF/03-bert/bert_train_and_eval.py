from config import Config
from data_preprocessing import build_dataloader
from bert_model import BertClassifierModel
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

config = Config()

def eval_model(model):
    # 1- 加载验证集
    """
        为什么训练的使用shuffle为True，评估的时候为False？
        答：训练的时候设置为True，会打散数据，能够让模型训练更加充分
            评估的时候设置为False，不会打散数据，那么评估结果比较稳定
    """
    dataloader = build_dataloader(datapath=config.dev_datapath, shuffle=False)

    # 2- 预测结果存储变量
    all_pred_result = []    # 存储所有的预测结果
    all_true_result = []    # 存储所有的真实结果

    # 3- 切换模式
    # 如果设置为train()训练模式，那么整个过程不是评估过程，而是将验证集当成训练集使用
    model.eval()

    # 4- 模型评估
    with torch.no_grad():
        for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(tqdm(dataloader),start=1):
            # 4.1- 将数据发送到指定设备
            input_ids = input_ids.to(device=config.device)
            attention_mask = attention_mask.to(device=config.device)
            token_type_ids = token_type_ids.to(device=config.device)
            labels = labels.to(device=config.device)

            # 4.2- 前向传播
            pred_output = model(input_ids,attention_mask,token_type_ids)

            # 4.3- 得到预测类别id
            pred_index = torch.argmax(pred_output,dim=-1)

            # 4.4- 记录预测和真实结果
            # 如果GPU的资源够，那么这里可以不用写cpu()。如果写上，表示将数据从GPU上迁移到CPU上进行处理，能够节约GPU的资源
            all_pred_result.extend(pred_index.cpu().tolist())
            all_true_result.extend(labels.cpu().tolist())

    # 5- 计算分类问题的评估指标
    """
        注意：目前项目是多分类问题，因此average不能给binary。
        macro：多分类的场景，样本分布均衡
        micro：多分类的场景，样本分布不均衡
    """
    f1score = f1_score(all_true_result,all_pred_result,average="macro")
    accuracy = accuracy_score(all_true_result,all_pred_result)
    precision = precision_score(all_true_result,all_pred_result,average="macro")
    recall = recall_score(all_true_result,all_pred_result,average="macro")

    return f1score, accuracy, precision, recall

def train_and_eval():
    # 1- 加载数据
    dataloader = build_dataloader(datapath=config.train_datapath,shuffle=True)

    # 2- 创建实例对象
    # 2.1- 模型实例对象
    model = BertClassifierModel().to(device=config.device)
    # 2.2- 优化器实例对象
    optim = torch.optim.AdamW(params=model.parameters(),lr=5e-5)
    # 2.3- 损失函数实例对象
    loss = nn.CrossEntropyLoss()

    # 3- 切换模式
    model.train()

    # 4- 模型训练
    epochs = 1
    best_f1score = 0.0 # F1值历史最高分

    for epoch in range(epochs):

        total_loss = 0.0 # 每个轮次中，总损失值

        for i,(input_ids,attention_mask,token_type_ids,labels) in enumerate(tqdm(dataloader),start=1):
            # 4.1- 将数据发送到指定设备
            input_ids = input_ids.to(device=config.device)
            attention_mask = attention_mask.to(device=config.device)
            token_type_ids = token_type_ids.to(device=config.device)
            labels = labels.to(device=config.device)

            # 4.2- 前向传播
            pred_output = model(input_ids,attention_mask,token_type_ids)

            # 4.3- 计算损失值
            loss_value = loss(pred_output,labels)
            total_loss += loss_value

            # 4.4- 固定代码
            optim.zero_grad()           # 梯度清零
            loss_value.sum().backward() # 反向传播
            optim.step()                # 更新参数


            # 4.5- 每隔100个批次或者最后一个批次，对模型进行一次评估
            if i%100==0 or i==len(dataloader):
                # 4.5.1- 调用评估函数
                f1score, accuracy, precision, recall = eval_model(model)
                print(f"第{epoch+1}轮次，第{i}批次，f1score={f1score}，accuracy={accuracy}，precision={precision}，recall={recall}")

                # 4.5.2- 如果当前的f1score超过了历史最高分，那么将当前的模型保存下来
                if f1score>best_f1score:
                    torch.save(model.state_dict(),config.save_model)

                    # 更新历史最高分
                    best_f1score = f1score

                # 4.5.3- 将模式切回训练模式
                model.train()

if __name__ == '__main__':
    train_and_eval()