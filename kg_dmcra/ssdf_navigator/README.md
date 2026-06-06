# SSDF-Navigator：DongYuan 的主动信息采集引擎

**SSDF-Navigator** 在 **DongYuan** 医学专家系统的 **KG-DMCRA**（知识接地双模块协同推理架构）中，实例化了 **DK-AIA**（领域知识接地的主动信息采集）范式，作为主动信息采集引擎。它是一个基于 Transformer 的轻量级策略模型，从真实医患对话中训练得到，用于中西医结合脾胃病诊疗中的多轮主动问诊。

> **📌 本模块是 [DongYuan](https://github.com/your-org/DongYuan) 项目的一部分。**  
> 完整系统概述请参考[根目录 README](../../README.zh-CN.md)。

---

## DK-AIA 的四个设计原则

SSDF-Navigator 遵循 DK-AIA 提出的四个可迁移设计原则，作为临床复杂医学专家系统中主动信息采样的通用范式。

### 1. 本体嵌入的状态抽象

原始交互历史以非结构化自然语言表达，语义相似的输入可能呈现高度多样的词汇和句法形式。为此，DK-AIA 将交互轨迹映射到一个**结构化状态空间** $\mathcal{S} = \{s_1, s_2, \dots, s_N\}$，其中每个状态对应一个标准化的领域信息单元。

对于一个包含 $t$ 个采集步骤的交互回合，原始历史被映射为状态序列 $\mathbf{s} = [s_0, s_1, \dots, s_t]$。使用**长度为 $L$ 的滑动窗口**构建决策状态：

$$\text{state}_t = [s_{t-L+1}, \dots, s_t]$$

策略定义为：

$$a_t \sim \pi(\cdot \mid \text{state}_t)$$

其中 $a_t = s_{t+1} \in \mathcal{S}$ 是下一个要采集的标准化信息单元。

### 2. 信息增益驱动的奖励塑形

**状态转移概率**通过离线历史数据结合拉普拉斯平滑进行估计：

$$P(y \mid x) = \frac{C(x, y) + \alpha}{C(x) + \alpha \cdot N}$$

其中 $C(x, y)$ 是状态 $x$ 与 $y$ 的共现频率，$C(x)$ 是状态 $x$ 的总频率，$N = |\mathcal{S}|$ 为状态空间大小，$\alpha$ 为拉普拉斯平滑因子。

**归一化条件熵**量化了查询症状 $x$ 后的剩余不确定性：

$$\tilde{H}(Y \mid x) = \frac{H(Y \mid x)}{\log N} = -\frac{1}{\log N} \sum_{y} P(y \mid x) \log P(y \mid x)$$

**信息增益奖励**：

$$R_{\text{info}}(a) = 1 - \tilde{H}(Y \mid a)$$

**重复因子** $\lambda(a, \mathcal{H})$ 在惩罚冗余的同时允许必要的关键症状回访：

$$\lambda(a, \mathcal{H}) = 
\begin{cases} 
0.3 & \text{即时重复（与前一次查询相同）} \\
1.5 & \text{延迟回访（非连续的关键症状）} \\
1.0 & \text{中性（新症状查询）}
\end{cases}$$

**最终即时奖励**（缩放以匹配 BC 损失量级）：

$$r_t = R_{\text{info}}(a) \cdot \lambda(a, \mathcal{H}) \cdot K, \quad K = 2.0 / 1.5$$

### 3. 混合模仿-强化策略学习

提出两种混合训练方案：

#### Reward-Weighted Behavior Cloning (RWBC)

通过即时奖励对行为克隆损失进行重新加权，更加重视与高信息增益相关的专家动作：

$$\mathcal{L}_{\text{RWBC}}(\theta) = -\frac{1}{B} \sum_{i=1}^{B} r_t \cdot \log \pi_{\theta}(a_t^* \mid X_t)$$

对应 `--fusion_mode mul`。

#### Reward-Added Behavior Cloning (RABC)

三个项的加权和——行为克隆损失、策略梯度损失和熵正则化：

$$\mathcal{L}_{\text{RABC}}(\theta) = \beta_1 \mathcal{L}_{\text{BC}} + \beta_2 \mathcal{L}_{\text{PG}} + \eta \mathcal{L}_{\text{Ent}}$$

其中：
- $\mathcal{L}_{\text{BC}}(\theta) = -\frac{1}{B} \sum \log \pi_{\theta}(a_t^* \mid X_t)$ — 监督式行为克隆
- $\mathcal{L}_{\text{PG}}(\theta) = -\frac{1}{B} \sum A_t \cdot \log \pi_{\theta}(a_t \mid X_t)$ — 策略梯度，优势函数 $A_t = r_t - b$
- $\mathcal{L}_{\text{Ent}}(\theta) = -\sum_a \pi_{\theta}(a \mid \text{state}) \log \pi_{\theta}(a \mid \text{state})$ — 熵正则化促进探索
- $\beta_1, \beta_2$ — 平衡权重；$\eta$ — 熵系数

对应 `--fusion_mode add`。

### 4. 双模块候选协调

为了将主动采集策略与 SSDF-Core（深度诊断推理引擎）在 KG-DMCRA 中进行整合：

1. 结构化历史队列 $\mathcal{H}_0$ 初始为空，首次交互由 SSDF-Core 生成
2. 从第二个时间步开始，仅当上一轮交互可映射到结构化状态空间且 $\mathcal{H}$ 非空时，SSDF-Navigator 才被激活
3. SSDF-Navigator 预测概率分布并将 **top-$K$ 预测**转换为标准化候选动作：$\mathcal{A}_{\text{ia}} = \{a_1, \dots, a_K\}$
4. SSDF-Core 基于完整原始交互历史独立生成自然语言候选，映射为 $a_{\text{core}}$
5. SSDF-Core 从**组合候选集** $\mathcal{A}_{\text{ia}} \cup \{a_{\text{core}}\}$ 中根据当前上下文和推理路径选择下一动作
6. 问诊在**信息饱和**（新问题反复映射到历史队列中已有的症状）或达到 **30 轮对话上限**时终止

---

## 在中西医结合脾胃病诊疗中的实例化

DK-AIA 被实例化为 **SSDF-Navigator**（如图 4 所示），一个用于多轮主动问诊的轻量级策略模型：

- **架构**：基于 Transformer 的编码器-分类器
- **输入**：结构化的症状历史队列（最近 $L$ 个症状）
- **输出**：83 个标准化核心症状的概率分布
- **标准化本体**：根据国家标准《中医临床诊疗术语·第 2 部分：证候》进行映射，涵盖脾胃病的核心症状，包括胃痛、胃胀、胃脘不适、反酸、嗳气、纳少等
- **训练数据**：从 **SSDF-Dialogue** 的脾胃病子集中提取医生问诊问题对应的症状；每个对话案例转换为症状状态序列；通过固定长度滑动窗口提取状态-动作转移样本
- **优化**：使用 RWBC（式 10）或 RABC（式 11）在转移样本上进行训练

---

## 快速开始

### 环境要求

```bash
Python 3.10+
PyTorch ≥ 2.0
```

### 训练

```bash
cd kg_dmcra/ssdf_navigator

# RWBC 模式（Reward-Weighted Behavior Cloning）
python train_navigator.py \
  --question_file ./data/question_cleaned.csv \
  --symptom_file ./data/symptoms.csv \
  --fusion_mode mul \
  --max_seq_len 10 \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-4 \
  --save_path ./models/transformer_policy.pth \
  --gpu 0

# RABC 模式（Reward-Added Behavior Cloning）
python train_navigator.py \
  --fusion_mode add \
  --lambda_supervised 1.0 \
  --lambda_reinforce 0.5 \
  --entropy_coef 0.01 \
  # ... 其他参数同上
```

### 评估

```bash
python evaluate_navigator.py \
  --question_file ./data/question_cleaned.csv \
  --symptom_file ./data/symptoms.csv \
  --model_path ./models/transformer_policy.pth \
  --topk 5 \
  --eval_split val \
  --gpu 0
```

### Navigator API 服务部署

```bash
cd ssdf_bench/mpc_evaluation
python navigator_api_service.py \
  --port 8999 \
  --model_path /path/to/transformer_policy.pth \
  --question-file ./data/question_cleaned.csv \
  --symptom-file ./data/symptoms.csv
```

---

## 引用

如您的研究中使用 SSDF-Navigator，请引用 DongYuan 论文：

```bibtex
@article{dongyuan2026dongyuan,
  title = {DongYuan: An LLM-Based Medical Expert System for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis},
  author = {Your Authors},
  journal = {Expert Systems with Applications},
  year = {2026},
  note = {Submitted for review}
}
```
