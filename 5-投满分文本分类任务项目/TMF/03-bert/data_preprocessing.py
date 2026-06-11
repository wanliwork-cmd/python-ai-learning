import torch
from tqdm import tqdm
from config import Config
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertConfig

# 1- 公共变量
config = Config()

bert_tokenizer = BertTokenizer.from_pretrained(config.bert_path)
bert_model = BertModel.from_pretrained(config.bert_path)
bert_config = BertConfig.from_pretrained(config.bert_path)

def load_raw_file(datapath):
    """
    加载并处理原始文件
    :param datapath: 原始文件路径
    :return: 处理后的文件，新闻标题string，目标值是int。格式：[(新闻标题,目标值),(新闻标题,目标值)...]
    """

    # 1- 读取原始文件内容
    with open(datapath,mode="r",encoding="UTF-8") as f:
        lines = f.readlines()

    # 2- 循环遍历，处理每条样本
    result_list = []    # 返回结果
    for line in tqdm(lines,desc="处理文件中"):
        # 2.1- 空行处理和判断
        line = line.strip()
        if line=="":
            continue

        # 2.2- 每行数据拆解为新闻标题和目标值
        title,label = line.split("\t")

        # 【可选】健壮性代码
        """
            只要是有数据类型转换的地方，基本都有健壮性代码
        """
        if not label.isdigit():
            print(f"label的数据内容不合法，值是{label}")
            continue

        # 2.3- 存储到列表中
        result_list.append((title,int(label)))

    return result_list

# 自定义数据集
class NewsDataset(Dataset):
    def __init__(self,data_list):
        # 设置属性值
        self.data_list = data_list

        # 样本条数
        self.sample_cnt = len(data_list)

    def __len__(self):
        return self.sample_cnt

    def __getitem__(self, index):
        # 1- 处理索引为负数；越界的问题
        index = min(max(index,0), self.sample_cnt - 1)

        # 2- 获得索引对应的样本
        title, label = self.data_list[index]

        return title,label

# 对每个批次的数据进行特定的处理。
def collate_fn(batch_data):
    # 1- 将每个批次中新闻标题组织成一个容器；label目标值也单独组织成一个容器
    """
    zip(*)处理过程如下：
        输入数据：[('近期新盘推荐 通州纯新别墅本周开盘', 1), ('陕西退休教师嫌弃精神病女儿将其勒死被捕', 5)]
        输出数据：[('近期新盘推荐 通州纯新别墅本周开盘', '陕西退休教师嫌弃精神病女儿将其勒死被捕'),     (1, 5)]
    """
    # print(f"输入数据：{batch_data}")
    titles_tup, labels_tup = zip(*batch_data)
    # print(f"输出数据：{list(zip(*batch_data))}")

    # 2- 对新闻标题进行分词处理，得到词索引张量
    data_tensor = bert_tokenizer.batch_encode_plus(
        titles_tup,
        max_length=config.max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # print(data_tensor)

    # 3- 解析分词后的结果
    input_ids = data_tensor.input_ids
    attention_mask = data_tensor.attention_mask

    # 可以不要下面的内容。因为上面是对句子一条条处理的，因此token_type_ids中的值全都是0。
    # token_type_ids表示词来自哪条句子
    token_type_ids = data_tensor.token_type_ids

    # 4- 转成张量并返回
    labels = torch.tensor(labels_tup,dtype=torch.long)

    return input_ids,attention_mask,token_type_ids,labels


# 自定义数据加载器
def build_dataloader(datapath,shuffle=False):
    # 1- 加载原始文件：转成[(新闻标题, 目标值), (新闻标题, 目标值)...]
    data_list = load_raw_file(datapath)

    # 2- 创建自定义数据集实例对象
    dataset = NewsDataset(data_list)

    # 3- 创建自定义Dataloader数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return dataloader

if __name__ == '__main__':
    # result_list = load_raw_file(config.dev_datapath)
    # print(result_list[:10])

    # for input_ids,attention_mask,labels in build_dataloader(config.train_datapath,True):
    for input_ids,attention_mask,labels in build_dataloader(config.dev_datapath,False):
        print("input_ids-->",input_ids)
        print("attention_mask-->",attention_mask)
        print("labels-->",labels)
        break



