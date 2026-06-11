from config import Config
from bert_model import BertClassifierModel
from transformers import BertTokenizer
import torch

# 1- 加载训练好的模型
config = Config()

model = BertClassifierModel().to(config.device)
# 注意：不要用变量接受返回结果
model.load_state_dict(torch.load(config.save_model))
tokenizer = BertTokenizer.from_pretrained(config.bert_path)

model.eval()

# 2- 预测函数
def predict(news_data):
    # news_data传递进来的数据格式要求：{"title":新闻标题内容}

    # 1- 获取传递过来的新闻标题
    title = news_data["title"]

    # 2- 分词处理得到词索引
    data_tensor = tokenizer.batch_encode_plus(
        [title],
        max_length=config.max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    print(data_tensor)
    # 3- 取分词后的内容
    input_ids = data_tensor.input_ids.to(device=config.device)
    attention_mask = data_tensor.attention_mask.to(device=config.device)
    token_type_ids = data_tensor.token_type_ids.to(device=config.device)

    # 4- 预测
    with torch.no_grad():
        pred_output = model(input_ids, attention_mask,token_type_ids)
        pred_index = torch.argmax(pred_output,dim=-1)
        # 通过预测类别索引获得类别名称
        pred_classname = config.classname_list[pred_index]

    # # 5- 返回预测结果
    news_data["pred_class"] = pred_classname
    return news_data

if __name__ == '__main__':
    print(predict({"title": "体验2D巅峰 倚天屠龙记十大创新概览"}))

