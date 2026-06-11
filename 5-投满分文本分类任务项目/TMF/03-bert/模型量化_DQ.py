# 注意：DQ推荐在CPU环境下运行
from config import Config
from bert_model import BertClassifierModel
import torch
from bert_train_and_eval import eval_model

if __name__ == '__main__':
    # 1- 【了解】查看支持的量化引擎
    """
        量化引擎：
            none：没有硬件加速
            onednn：英特尔深度学习
            x86：x86架构的服务器进行优化
            fbgemm：FaceBook的量化计算引擎
    """
    print("查看支持的量化引擎：",torch.backends.quantized.supported_engines)

    # 2- 【可选】可以手动设置使用的量化引擎
    # 如果不手动设置，那么代码内部会自动根据你硬件的情况进行自动选择
    torch.backends.quantized.engine="onednn"

    # 3- 加载配置
    config = Config()

    # 4- 评估量化前的模型效果
    # 4.1- 创建模型实例对象
    model = BertClassifierModel()
    # 4.2- 加载训练好的模型
    # map_location：将模型的参数等信息全部加载到CPU上
    model.load_state_dict(torch.load(config.save_model, map_location="cpu"))
    # 4.3- 设置模式为评估模式
    model.eval()
    # 4.4- 评估模型
    f1score, accuracy, precision, recall = eval_model(model)
    print(f"量化前的指标：f1score={f1score}，accuracy={accuracy}，precision={precision}，recall={recall}")


    # 5- 模型量化：DQ的实现方式。量化就是降低参数的数据精度
    """
        参数解释：
            model：指定要量化的模型
            qconfig_spec：指定对模型的什么地方进行参数类型调整。参数类型是Set集合
                qconfig_spec={torch.nn.Linear}意思是对BERT模型的所有线性层进行参数类型调整
                回顾：多头自注意力中有4个线性层、前馈网络等
            dtype：将参数的数据类型降低到什么样的新数据类型。
                qint8和int8的区别：qint8是专门为量化涉及的一种新数据类型
    """
    quantization_model = torch.quantization.quantize_dynamic(model=model,qconfig_spec={torch.nn.Linear},dtype=torch.qint8)

    # 6- 评估量化后的模型
    quantization_model.eval()
    f1score, accuracy, precision, recall = eval_model(quantization_model)
    print(f"量化后的指标：f1score={f1score}，accuracy={accuracy}，precision={precision}，recall={recall}")

    # 7- 打印量化后的模型内部信息
    print(quantization_model)

    # 8- 保存量化后的模型
    torch.save(quantization_model,"save_model/quantization_model.pkl")



