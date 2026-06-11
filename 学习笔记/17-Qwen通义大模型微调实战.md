# 17 - 基于Qwen通义大模型微调实战（LoRA/QLoRA + DeepSpeed + LLaMA-Factory + vLLM）

> **日期**：2026年1月15日 ~ 1月22日  
> **核心技术**：Qwen通义千问、LoRA/QLoRA微调、DeepSpeed ZeRO-3、LLaMA-Factory、vLLM部署  
> **项目目标**：基于Qwen大模型进行文本摘要任务的微调训练与高效部署

---

## 一、项目介绍

### 1.1 项目背景

随着信息爆炸时代的到来，如何高效提取文本关键信息成为重要需求。本项目基于**通义千问（Qwen）**大模型，对文本中的关键信息进行自动抽取并生成简洁摘要。

**应用场景**：
- **新闻媒体**：自动生成新闻要点，提升内容分发效率
- **金融分析**：从财报、研报中提取关键指标与风险事件
- **医疗健康**：解析病历、文献，生成患者病情摘要
- **法律文书**：快速抽取案件核心信息，辅助法律检索
- **企业知识库**：自动化归档会议记录、技术文档

### 1.2 模型选型：通义千问（Qwen）

- 支持长文本输入（最高32k tokens），适配复杂语义场景
- 多语言能力与领域泛化性强，覆盖中英文混合文本处理
- 开源生态完善，便于二次开发与优化
- 可选模型：Qwen-1.8B、Qwen-7B、Qwen-14B 等

---

## 二、DeepSpeed介绍

### 2.1 什么是DeepSpeed

DeepSpeed 是由**微软**开发的开源库，专为大规模模型训练设计。核心能力：
- 降低训练超大规模模型的复杂性和资源需求
- 通过**模型并行化、梯度累积、动态精度缩放**等技术加速训练
- 提供分布式训练管理、内存优化和模型压缩等辅助工具
- 基于 PyTorch 构建，只需简单修改即可迁移

### 2.2 安装与配置

```bash
pip install deepspeed
# 验证安装
deepspeed --version
```

### 2.3 DeepSpeed配置文件（ds_config_zero2.json）

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": { "enabled": "auto" },
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto" }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": { "warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto" }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": { "device": "none", "pin_memory": true },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

### 2.4 ZeRO优化阶段对比

| 阶段 | 名称 | 优化内容 | 显存节省 |
|------|------|----------|----------|
| ZeRO-1 | 优化器状态分区 | 将优化器状态分到多个GPU | ~4x |
| ZeRO-2 | + 梯度分区 | + 梯度也分区存储 | ~8x |
| ZeRO-3 | + 参数分区 | + 模型参数也分区 | ~N×GPU数 |

**ZeRO-3** 最激进，但要注意：ZeRO-3 与 LoRA 在非 chat 模型上不兼容。

---

## 三、自定义训练脚本（finetune.py）

### 3.1 参数定义（四大配置类）

```python
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")

@dataclass
class DataArguments:
    data_path: str = field(default=None)        # 训练数据路径
    eval_data_path: str = field(default=None)    # 验证数据路径
    lazy_preprocess: bool = False                # 是否懒惰预处理

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")    # 优化器
    model_max_length: int = field(default=8192)  # 最大序列长度
    use_lora: bool = False                       # 是否使用LoRA

@dataclass
class LoraArguments:
    lora_r: int = 64                             # LoRA秩
    lora_alpha: int = 16                         # 缩放因子
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]  # Qwen特定目标模块
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False                         # 是否使用QLoRA
```

### 3.2 数据预处理（Qwen Chat格式）

Qwen 使用特殊的对话格式标记：

```python
def preprocess(sources, tokenizer, max_len, system_message="You are a helpful assistant."):
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    
    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    
    for source in sources:
        input_id, target = [], []
        
        # 系统消息
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        
        # 遍历对话中的每个句子
        for sentence in source:
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            
            if role == '<|im_start|>user':
                # 用户部分不计算loss
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                # 助手部分计算loss
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                          _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
            target += _target
        
        # 填充至max_len
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
```

**Qwen对话格式**：
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
用户消息<|im_end|>
<|im_start|>assistant
助手回复<|im_end|>
```

**损失计算策略**：
- `IGNORE_TOKEN_ID`（-100）：用户消息和系统消息不计算损失
- 只对 **assistant 回复内容** 计算损失（角色标签本身也忽略）

### 3.3 数据集类

```python
class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_len):
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
    
    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
```

### 3.4 DeepSpeed ZeRO-3 兼容处理

```python
def maybe_zero_3(param):
    """处理ZeRO-3分区参数，确保可从GPU复制到CPU"""
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param
```

ZeRO-3 将模型参数分区到多个GPU上，保存模型时需要先 `Gather` 收集完整参数。

### 3.5 训练主函数

```python
def train():
    # 解析命令行参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    # 加载模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        quantization_config=GPTQConfig(bits=4, disable_exllama=True) 
            if training_args.use_lora and lora_args.q_lora else None,
        trust_remote_code=True,
    )
    
    # 加载tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id  # Qwen用eod作为pad
    
    # 配置LoRA
    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,          # 64
            lora_alpha=lora_args.lora_alpha,  # 16
            target_modules=lora_args.lora_target_modules,  # ["c_attn","c_proj","w1","w2"]
            lora_dropout=lora_args.lora_dropout,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,  # 非chat模型保存wte和lm_head
        )
        
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(model)  # 4-bit量化准备
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 初始化Trainer并训练
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
```

---

## 四、LLaMA-Factory微调框架

### 4.1 简介

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 是一个简单易用且高效的大模型训练框架，支持上百种大模型。

**核心特性**：
- **模型支持**：LLaMA、Qwen、Yi、Gemma、Baichuan、ChatGLM、Phi 等
- **训练算法**：预训练、指令微调、奖励模型、PPO/DPO/KTO/ORPO 训练
- **精度支持**：16-bit全参数、冻结微调、LoRA、QLoRA（2/4/6/8-bit）
- **优化算法**：GaLore、BAdam、DoRA、LongLoRA、LoRA+ 等
- **加速算子**：FlashAttention-2、Unsloth
- **推理引擎**：Transformers、vLLM
- **实验面板**：LlamaBoard、TensorBoard、Wandb、MLflow

### 4.2 安装

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
# 验证安装
llamafactory-cli version
```

### 4.3 LoRA微调配置（qwen2-7b-lora-sft.yaml）

```yaml
### model
model_name_or_path: Qwen/Qwen2-7B-Instruct

### method
stage: sft                          # 监督微调阶段
do_train: true
finetuning_type: lora               # LoRA微调
lora_target: all                    # 对所有线性层应用LoRA
lora_rank: 16
lora_alpha: 16
lora_dropout: 0.05

### dataset
dataset: qwen_train_data
template: qwen                      # 使用Qwen对话模板
cutoff_len: 1024                    # 最大序列长度
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/qwen2-7b/lora/sft
logging_steps: 100
save_steps: 100
plot_loss: true                     # 绘制损失曲线

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16     # 梯度累积16步
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine           # 余弦退火
warmup_ratio: 0.1
bf16: true                          # 使用bfloat16精度

### eval
val_size: 0.1                       # 10%数据作为验证集
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

### 4.4 训练命令

```bash
# LoRA微调
llamafactory-cli train qwen2-7b-lora-sft.yaml

# QLoRA微调（4-bit量化）
llamafactory-cli train qwen2-7b-qlora-sft.yaml

# 全量微调 + DeepSpeed
deepspeed --num_gpus 2 src/train_bash.py \
    --deepspeed deepspeed/ds_config_zero2.json \
    qwen2-7b-full-sft.yaml

# 合并LoRA权重
llamafactory-cli export qwen2-7b-merge-lora.yaml
```

### 4.5 自定义训练脚本 vs LLaMA-Factory 对比

| 对比项 | 自定义finetune.py | LLaMA-Factory |
|--------|-------------------|---------------|
| 灵活性 | 高，可完全自定义 | 中，通过YAML配置 |
| 代码量 | 多（~400行） | 少（YAML配置） |
| 支持模型 | 手动适配 | 内置上百种 |
| 训练算法 | 需手动实现 | 内置SFT/PPO/DPO等 |
| 适用场景 | 需要深度定制 | 快速实验和部署 |

---

## 五、Qwen推理示例（inference.py）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-1_8B-Chat", device_map="auto", trust_remote_code=True
).eval()

# 配置生成参数
model.generation_config = GenerationConfig.from_pretrained(
    "Qwen/Qwen-1_8B-Chat", trust_remote_code=True
)

# 多轮对话
response, history = model.chat(tokenizer, "3.9和3.11，哪个值比较大", history=None)
print(response)

# 第二轮对话（传入history）
response, history = model.chat(tokenizer, "给我讲一个创业故事", history=history)

# 第三轮对话
response, history = model.chat(tokenizer, "给故事起个标题", history=history)
```

**精度选择**：
- `bf16=True`：A100、H100、RTX3060/3070 推荐
- `fp16=True`：V100、P100、T4 推荐
- 默认自动选择

---

## 六、vLLM高效部署

### 6.1 vLLM简介

vLLM 是由**加州大学伯克利分校**开发的高性能大模型推理框架，核心优势：
- **PagedAttention**：灵感来自操作系统的虚拟内存分页管理
  - 动态分配KV Cache显存，提升显存利用率
  - 像操作系统管理内存页一样管理KV Cache
- 高吞吐量的推理服务
- 兼容OpenAI API格式

### 6.2 推理过程

LLM推理分为两个阶段：
1. **Prefill阶段**：处理输入的prompt，生成KV Cache
2. **Decode阶段**：根据prefill结果，一个token一个token地生成response

**KV Cache问题**：
- 随prompt数量增多和序列变长，KV Cache对GPU显存造成巨大压力
- 输出序列长度无法预先知道，难以提前分配存储空间
- 例如：13B模型在A100 40GB上推理时，KV Cache占用显著

### 6.3 安装与使用

```bash
pip install vllm
```

**通过OpenAI兼容API调用**：
```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

chat_response = client.chat.completions.create(
    model="/path/to/Qwen/model/",
    messages=[
        {"role": "user", "content": "总结下面这段文本的摘要..."},
    ]
)
print("Chat response:", chat_response)
```

---

## 七、MCP协议（扩展知识）

讲义还介绍了 **MCP（Model Context Protocol）** 协议：
- 标准化的模型上下文协议
- 支持百炼平台接入MCP
- Qwen模型可直接接入MCP生态
- Qwen-Agent 框架的应用实践

---

## 八、关键技术总结

| 技术 | 用途 | 关键配置 |
|------|------|----------|
| Qwen | 基础大模型 | 1.8B/7B/14B 可选 |
| LoRA | 参数高效微调 | r=64, alpha=16, targets=[c_attn,c_proj,w1,w2] |
| QLoRA | 4-bit量化微调 | bits=4, GPTQConfig |
| DeepSpeed ZeRO-2/3 | 分布式训练优化 | stage 2/3, 梯度累积 |
| LLaMA-Factory | 训练框架 | YAML配置，一键训练 |
| vLLM | 高效推理部署 | PagedAttention, OpenAI API兼容 |
| Gradient Checkpointing | 显存优化 | 用计算换显存 |

---

## 九、依赖环境

```bash
# 核心依赖
pip install transformers deepspeed peft accelerate
# LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
pip install -e ".[torch,metrics]"
# vLLM
pip install vllm
# OpenAI客户端（用于测试）
pip install openai>=1.5.0
```
