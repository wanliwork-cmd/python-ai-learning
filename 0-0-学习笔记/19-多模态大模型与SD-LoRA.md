# 19 - 多模态大模型与Stable Diffusion LoRA训练

> **日期**：2026年1月27日 ~ 2月3日  
> **核心技术**：AIGC、Stable Diffusion、扩散模型、LoRA训练（图像生成）  
> **课程形式**：理论讲义（PDF），实践操作（LoRA训练脚本）

---

## 一、AIGC概述

### 1.1 什么是AIGC

AIGC（AI Generated Content）即**人工智能生成内容**，是指利用AI技术自动生成文本、图像、音频、视频等多模态内容。

**AIGC发展里程碑**：
- 文本生成：GPT系列、LLaMA、Qwen 等大语言模型
- 图像生成：DALL·E、Midjourney、Stable Diffusion
- 视频生成：Sora、Runway
- 音频生成：MusicLM、AudioCraft

### 1.2 AIGC核心任务

| 任务类型 | 代表模型 | 应用场景 |
|---------|---------|---------|
| 文生图 | Stable Diffusion, DALL·E | 创意设计、广告 |
| 图生图 | Stable Diffusion + ControlNet | 风格迁移、修复 |
| 文生视频 | Sora, Runway | 影视制作 |
| 文生音频 | MusicLM | 音乐创作 |

---

## 二、Stable Diffusion详解

### 2.1 扩散模型基本原理

扩散模型（Diffusion Model）的核心思想分为两个过程：

**前向扩散过程（加噪）**：
```
清晰图像 → 逐步加噪 → 纯噪声
x_0 → x_1 → x_2 → ... → x_T ≈ N(0,I)
```
- 每一步向图像添加少量高斯噪声
- 经过T步后，图像变成纯随机噪声

**反向去噪过程（生成）**：
```
纯噪声 → 逐步去噪 → 清晰图像
x_T → x_{T-1} → ... → x_1 → x_0
```
- 模型学习预测每一步添加的噪声
- 从纯噪声开始，逐步恢复出清晰图像

### 2.2 Stable Diffusion架构

Stable Diffusion 是**潜在扩散模型（Latent Diffusion Model）**，不在像素空间直接扩散，而是在**潜在空间**中进行：

```
文本输入 → CLIP文本编码器 → 文本嵌入
                                ↓
随机噪声 → UNet（在潜在空间中预测噪声）→ 去噪后的潜在表示 → VAE解码器 → 生成图像
                                ↑
                         时间步t
```

**三大核心组件**：
1. **VAE（变分自编码器）**：将图像压缩到低维潜在空间
2. **UNet**：在潜在空间中预测和去除噪声（核心去噪网络）
3. **CLIP文本编码器**：将文本描述转为嵌入向量，引导图像生成

### 2.3 关键参数

| 参数 | 说明 | 常用值 |
|------|------|--------|
| Steps | 去噪步数 | 20-50 |
| CFG Scale | 文本引导强度 | 7-12 |
| Sampler | 采样算法 | Euler a, DPM++ 2M |
| Resolution | 输出分辨率 | 512×512, 768×768 |
| Seed | 随机种子 | -1（随机） |

---

## 三、Stable Diffusion训练与部署

### 3.1 模型部署方式

1. **本地部署**：Stable Diffusion WebUI（AUTOMATIC1111）
   - 支持丰富的插件生态
   - 需要较强GPU（建议8GB+显存）

2. **云端部署**：
   - 腾讯云AI绘画服务
   - 趋动云等GPU云平台

3. **Diffusers库**：HuggingFace提供的Python API

### 3.2 LoRA在图像生成中的应用

LoRA（Low-Rank Adaptation）不仅可以用于文本大模型微调，在图像生成领域也有广泛应用：

**SD LoRA的作用**：
- 让模型学习特定的**画风、人物、概念**
- 只需少量图片（10-50张）即可训练
- 训练后的LoRA权重可与基础模型自由组合

### 3.3 LoRA训练工具

本项目使用 **sd-scripts**（kohya-ss开发）进行LoRA训练：

```
lora-scripts/
├── sd-scripts/          # 核心训练脚本
│   ├── library/         # 训练工具库
│   ├── train_network.py # LoRA训练入口
│   ├── fine_tune.py     # 全量微调脚本
│   └── finetune/        # 数据预处理工具
├── sd-models/           # 放置基础模型(.ckpt)
├── output/              # 训练输出（LoRA权重）
│   ├── glass-000002.safetensors
│   ├── glass-000004.safetensors
│   ├── glass-000006.safetensors
│   ├── glass-000008.safetensors
│   └── glass.safetensors  # 最终LoRA权重
└── huggingface/         # CLIP等预训练模型缓存
```

### 3.4 LoRA训练流程

**Step 1：准备训练数据**
- 收集10-50张目标风格/概念的图片
- 为每张图片编写描述文本（caption）
- 使用 BLIP 等模型自动生成caption
- 使用 WD14 Tagger 自动打标签

**Step 2：数据预处理**
```python
# finetune/make_captions.py - 使用BLIP生成图片描述
# finetune/tag_images_by_wd14_tagger.py - 自动打标签
# finetune/merge_captions_to_metadata.py - 合并元数据
# finetune/prepare_buckets_latents.py - 预计算VAE潜在表示
```

**Step 3：LoRA训练**
- 设置LoRA rank（秩）、学习率、训练步数
- 使用 Accelerate 进行分布式训练
- 训练过程中定期保存checkpoint

**Step 4：推理验证**
- 加载基础模型 + LoRA权重
- 使用特定触发词生成图像
- 调整LoRA权重强度观察效果

### 3.5 训练配置（Accelerate）

```yaml
# huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: fp16       # 使用半精度训练
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 3.6 CLIP文本编码器

项目使用 OpenAI 的 `clip-vit-large-patch14` 作为文本编码器：
- 将文本描述编码为768维嵌入向量
- 引导UNet生成符合描述的图像
- 与图像编码器共同训练，实现图文对齐

---

## 四、腾讯云AI绘画

讲义还介绍了腾讯云的AI绘画服务：
- 基于Stable Diffusion的云端AI绘画平台
- 支持文生图、图生图等多种功能
- 提供API接口供开发者调用
- 适合不想本地部署GPU环境的用户

---

## 五、文本LoRA vs 图像LoRA对比

| 对比项 | 文本大模型LoRA | Stable Diffusion LoRA |
|--------|---------------|----------------------|
| 目标模型 | LLM（GPT/Qwen/ChatGLM） | UNet（扩散模型） |
| 训练数据 | 文本问答对 | 图片+文本描述 |
| 应用 | 特定任务微调（分类/抽取/对话） | 特定风格/人物/概念学习 |
| 训练量 | 数百~数千条 | 10-50张图片 |
| 输出 | 文本 | 图像 |
| 核心原理 | 低秩矩阵分解（相同） | 低秩矩阵分解（相同） |

---

## 六、关键技术总结

| 技术 | 说明 |
|------|------|
| AIGC | AI生成内容，涵盖文本/图像/视频/音频 |
| 扩散模型 | 前向加噪+反向去噪的生成范式 |
| Stable Diffusion | 在潜在空间进行扩散的高效图像生成模型 |
| VAE | 图像压缩到潜在空间 |
| UNet | 核心去噪网络 |
| CLIP | 文本编码器，提供文本引导 |
| SD LoRA | 用少量图片学习特定风格/概念 |
| sd-scripts | kohya-ss开发的SD训练工具集 |
| BLIP | 自动生成图片描述(caption) |
| WD14 Tagger | 自动为图片打标签 |
