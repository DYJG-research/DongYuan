# DongYuan（东垣）：面向中西医结合脾胃病诊疗的基于LLM的医学专家系统

<p align="center">
  <img src="https://img.shields.io/badge/系统-DongYuan-blue.svg" alt="DongYuan">
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/许可证-Apache%202.0-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/专科-中西医结合-brightgreen.svg" alt="Integrative Medicine">
</p>

<p align="center">
  <b>DongYuan（东垣）</b>是一个基于大语言模型的医学专家系统，专为中西医结合脾胃病诊疗而设计。系统构建于三个逐层递进的层次之上——专家知识标注的数据基础、基于知识的双模块协同推理架构（KG-DMCRA）、以及过程-结果耦合评估基准（SSDF-Bench）——DongYuan 致力于解决在交互式诊断流程中需要同时推理中医辨证分型与西医疾病证据这一临床挑战。
</p>

---

## 📋 项目简介

大语言模型在医学诊断任务上已展现出性能提升，但将其转化为临床可用的专家系统在复杂的疾病特异性诊断场景中仍面临困难。这一挑战在中西医结合脾胃病诊疗中尤为突出——中医辨证分型与西医疾病证据需要在交互式诊断流程中被联合推理。

**DongYuan** 通过三个逐层递进的层次解决这一问题：

1. **专家知识标注的数据基础**——包含三个专门数据集：*SSDF-Syndrome*、*SSDF-Dialogue* 和 *SSDF-PD*
2. **基于知识的双模块协同推理架构（KG-DMCRA）**——深度诊断推理引擎（*SSDF-Core*）与主动信息采集引擎（*SSDF-Navigator*）协同工作
3. **SSDF-Bench**——过程-结果耦合评估基准，配备经过验证的 LLM-as-a-judge 协议

实验表明，**SSDF-Core** 在 **SSDF-Bench** 上优于 12 个主流基线模型，架构级分析证实了双模块协同的贡献。DongYuan 为临床复杂、异构知识的诊断场景中的医学专家系统提供了一种通用设计方向。

### 核心特性

- **中西医结合诊断**：在交互式诊断流程中联合推理中医辨证分型与西医疾病证据
- **三层架构**：数据基础 → 双模块协同推理 → 过程-结果耦合评估
- **基于知识的双模块协同（KG-DMCRA）**：深度诊断推理引擎（SSDF-Core）+ 主动信息采集引擎（SSDF-Navigator）
- **专家知识标注数据集**：SSDF-Syndrome、SSDF-Dialogue、SSDF-PD，经专家指导标准化构建
- **过程-结果耦合评估**：SSDF-Bench 通过经过验证的 LLM-as-a-judge 协议同时评估诊断结果与问诊过程
- **标准化症状空间**：83 个经过专家验证的脾胃病标准症状术语体系

---

## 🏗️ 系统架构

```
DongYuan/
├── kg_dmcra/                              # Knowledge-Grounded Dual-Module Collaborative Reasoning Architecture
│   ├── coordination/                      # KG-DMCRA 协同问诊
│   │   └── llm_navigator_coordination.py  #   SSDF-Core 与 SSDF-Navigator 的协同编排
│   ├── ssdf_core/                         # SSDF-Core：深度诊断推理引擎
│   │   ├── run_ssdf_core_sft_training.sh  #   监督微调（SFT）脚本
│   │   └── run_ssdf_core_dpo_training.sh  #   直接偏好优化（DPO）脚本
│   └── ssdf_navigator/                    # SSDF-Navigator：主动信息采集引擎
│       ├── policy_network.py              #   Transformer 策略网络
│       ├── behavior_cloning.py            #   混合 BC + RL 训练器
│       ├── data_loader.py                 #   离线强化学习数据加载
│       ├── train_navigator.py             #   导航模型训练入口
│       ├── evaluate_navigator.py          #   导航模型评估入口
│       └── data_construction/             #   训练数据构建
│           ├── extract_questions_from_chat_records.py
│           ├── map_questions_to_symptoms.py
│           ├── normalize_symptom_names.py
│           ├── split_conversation_segments.py
│           ├── extract_symptom_inventory.py
│           ├── filter_questions_by_mapping.py
│           ├── filter_short_dialogues.py
│           └── group_question_symptom_mapping_by_symptom.py
├── ssdf_bench/                            # SSDF-Bench：过程-结果耦合评估
│   ├── ssdf_bench.py                     # 基准测试主入口
│   ├── benchmark_core/                   # 核心评估模块
│   │   ├── data_loader.py               # 数据加载
│   │   ├── model_interface.py           # 统一模型接口（API & 本地）
│   │   ├── report_generator.py          # HTML 报告生成
│   │   └── utils.py                     # 工具函数
│   ├── evaluators/                      # 评估器
│   │   ├── multiple_choice_evaluator.py # 选择题评估
│   │   └── llm_judge_evaluator.py       # LLM-as-Judge 评估
│   ├── mpc_evaluation/                  # 多患者问诊（MPC）评估
│   │   ├── run_mpc_collaborative_evaluation.py # 协同问诊评估
│   │   ├── run_mpc_pure_llm_evaluation.py      # 纯 LLM 问诊评估
│   │   ├── llm_patient_simulator.py            # 患者模拟器
│   │   ├── navigator_api_service.py            # Navigator API 服务
│   │   └── evaluate_mpc.py                     # MPC 结果分析
│   ├── evaluation_prompts/              # 评估提示词模板
│   ├── configs/                         # 配置文件
│   └── scripts/                         # 部署与评估脚本
└── data_foundation/                     # 专家知识标注的数据基础
    ├── evaluation_data/                 # 评估数据集
    │   ├── sdt_data.jsonl               # SSDF-Syndrome：辨证论治案例
    │   ├── mpc_data.json                # SSDF-PD：多患者问诊数据
    │   └── other_task_data.json         # 选择题数据（中医、西医、伦理、安全）
    └── train_data/                      # 训练数据集
        ├── sft_data/                    # SFT 数据（COT、多轮对话、考试增强）
        └── dpo_data/                    # DPO 偏好数据
```

---

## 🔬 方法论

### 三层架构

DongYuan 由三个逐层递进的层次构成：

#### 第一层：专家知识标注的数据基础

数据基础的核心包含三个专门数据集，均在中医专家指导下标注构建：

| 数据集 | 描述 |
|---------|------|
| **SSDF-Syndrome** | 辨证论治案例，包含完整患者病历、舌诊脉诊、检查检验结果及金标准证型标签 |
| **SSDF-Dialogue** | 多轮医患问诊对话，映射到 83 个标准化症状，用于训练主动信息采集引擎 |
| **SSDF-PD** | 多患者问诊评估数据，覆盖多种脾胃疾病的结构化症状档案 |

#### 第二层：KG-DMCRA（基于知识的双模块协同推理架构）

DongYuan 的核心是 KG-DMCRA，它实例化了两个互补的引擎：

- **SSDF-Core（深度诊断推理引擎）**：在 Qwen3-14B 基座上经过 SFT + DPO 两阶段微调的大语言模型，具备中西医结合诊疗所需的综合临床知识。给定患者的症状史和检查结果，SSDF-Core 同时对中医辨证分型和西医疾病证据进行深度推理。

- **SSDF-Navigator（主动信息采集引擎）**：一个轻量级 Transformer 编码器-分类器策略网络，通过行为克隆（BC）与信息增益奖励驱动的离线强化学习（RL）混合训练范式，学习专家的递进式问诊策略，在多轮交互中主动采集缺失但关键的鉴别性症状信息。

两个引擎在闭环诊断流程中协同工作：

1. SSDF-Core 基于当前患者信息生成候选问诊问题
2. 将问题映射到标准化症状空间
3. SSDF-Navigator 评估当前症状状态，输出 top-K 症状推荐及其概率分布
4. KG-DMCRA 协调决策：若 SSDF-Core 的问诊与 Navigator 的高置信度推荐一致则继续；否则通过阈值引导的协商机制优化下一步问诊
5. 新的患者回答更新症状状态，循环继续直至获取足够信息进行诊断

##### SSDF-Navigator 训练

**问题定义**——将多轮问诊建模为序贯决策问题：
- **状态（State）**：最近 $L$ 个历史标准化症状构成的状态窗口 $state_t = [s_{t-L+1}, ..., s_t]$
- **动作（Action）**：专家下一步询问的标准症状 $a_t = s_{t+1}$
- **策略（Policy）**：学习 $\pi(a_t | state_t)$ 以模拟专家递进式问诊逻辑

**模型架构**——SSDF-Navigator 采用 Transformer 编码器-分类器设计（嵌入层 + 正弦位置编码 → Transformer 编码器 → 掩码平均池化 → 83 维分类头）。

**混合训练范式**——融合模仿学习与离线强化学习：

1. **信息增益奖励函数**：$R_{info}(a) = 1 - \tilde{H}(Y | a)$，其中 $\tilde{H}(Y | a)$ 是查询症状 $a$ 后的归一化条件熵，并引入重复惩罚因子 $\lambda(a, \mathcal{H})$

2. **Reward-Weighted Behavior Cloning（RWBC）**（`--fusion_mode mul`）：$\mathcal{L}_{RWBC}(\theta) = -\frac{1}{B} \sum r_t \cdot \log \pi_{\theta}(a_t^* | X_t)$——以奖励作为样本权重

3. **Reward-Added Behavior Cloning（RABC）**（`--fusion_mode add`）：$\mathcal{L}_{RABC}(\theta) = \beta_1 \mathcal{L}_{BC} + \beta_2 \mathcal{L}_{PG} + \mathcal{L}_{Ent}$——结合监督损失、策略梯度与熵正则化

#### 第三层：SSDF-Bench（过程-结果耦合评估）

SSDF-Bench 提供综合性评估框架，同时评估诊断结果与推理过程：

**辨证论治维度：**

| 维度 | 评估方法 | 描述 |
|---------|-----------|------|
| 证型 | 多选题（多选） | 辨证分型判断 |
| 病性 | 单选题 | 疾病虚实寒热性质 |
| 病位 | 多选题（多选） | 病变脏腑定位 |
| 治则治法 | 多选题（多选） | 治疗原则与方法选择 |
| 病因 | LLM-as-Judge | 发病原因分析 |
| 病机 | LLM-as-Judge | 病理机制阐释 |
| 治疗方法 | LLM-as-Judge | 具体治疗手段 |
| 注意事项 | LLM-as-Judge | 医嘱与调护建议 |
| CoT 完备性 | LLM-as-Judge | 患者信息利用的全面性 |
| CoT 准确性 | LLM-as-Judge | 推理过程中幻觉的检查 |

**附加评估类别：**

| 类别 | 描述 | 权重 |
|----------|------|---------|
| 中医药学 | 中医基础知识 | 20% |
| 西医药学 | 西医知识 | 20% |
| 医学伦理 | 医学伦理考量 | 10% |
| 安全评估 | 模型安全性与对齐 | 10% |

**多患者问诊（MPC）评估：** 使用 LLM 模拟患者，评估完整多轮问诊中的问诊效率、关键症状覆盖率和诊断准确率。

**评分体系：** 辨证论治（40%）+ 中医药学（20%）+ 西医药学（20%）+ 医学伦理（10%）+ 安全评估（10%）。选择题评分采用 Sp 指标：$Sp = |A \cap B| / (|A| + |\bar{A} \cap B|)$。

---

## 🚀 快速开始

### 环境要求

```bash
Python 3.10+
PyTorch ≥ 2.0
```

### 安装

```bash
git clone https://github.com/your-org/DongYuan.git
cd DongYuan
pip install -r requirements.txt
```

### 训练 SSDF-Navigator（主动信息采集引擎）

```bash
cd kg_dmcra/ssdf_navigator

# RWBC 模式（Reward-Weighted Behavior Cloning）
python train_navigator.py \
  --question_file ./question_cleaned.csv \
  --symptom_file ./symptoms.csv \
  --fusion_mode mul \
  --max_seq_len 10 \
  --batch_size 64 \
  --epochs 100 \
  --lr 1e-4 \
  --save_path transformer_policy_mul.pth \
  --gpu 0

# RABC 模式（Reward-Added Behavior Cloning）
python train_navigator.py \
  --fusion_mode add \
  --lambda_supervised 1.0 \
  --lambda_reinforce 0.5 \
  --entropy_coef 0.01 \
  # ... 其他参数同上
```

### 评估 SSDF-Navigator

```bash
python evaluate_navigator.py \
  --question_file ./question_cleaned.csv \
  --symptom_file ./symptoms.csv \
  --model_path transformer_policy_mul.pth \
  --topk 5 \
  --eval_split val \
  --gpu 0
```

### 部署 Navigator API 服务

```bash
cd ssdf_bench/mpc_evaluation
python navigator_api_service.py \
  --port 8999 \
  --model_path /path/to/transformer_policy_mul.pth
```

### 训练 SSDF-Core（深度诊断推理引擎）

**阶段一——监督微调（SFT）：**
```bash
cd kg_dmcra/ssdf_core
bash run_ssdf_core_sft_training.sh
```

**阶段二——直接偏好优化（DPO）：**
```bash
bash run_ssdf_core_dpo_training.sh
```

### 运行 KG-DMCRA 协同问诊

```bash
cd ssdf_bench/mpc_evaluation

# 纯 LLM 基线（仅 SSDF-Core）
python run_mpc_pure_llm_evaluation.py \
  --core_model "your-model" \
  --core_model_base_url "http://localhost:8000/v1"

# KG-DMCRA 协同评估（SSDF-Core + SSDF-Navigator）
python run_mpc_collaborative_evaluation.py \
  --use_ssdf_navigator \
  --ssdf_core_model "your-model" \
  --ssdf_core_model_base_url "http://localhost:8000/v1" \
  --navigator_url "http://localhost:8999/predict" \
  --navigator_topk 3
```

### 运行 SSDF-Bench 基准评估

```bash
cd ssdf_bench

python ssdf_bench.py \
  --model_type api \
  --api_url "http://localhost:8000/v1" \
  --model_name "your-model" \
  --api_key "your-api-key" \
  --output_dir "results/your-model-run"
```

---

## 📁 数据格式

### SSDF-Syndrome：辨证论治案例

```json
{
  "id": "case_001",
  "instruction": "患者完整症状描述，包含舌象、脉象、检查检验结果...",
  "output": {
    "证型": "脾胃虚弱",
    "证型答案": "A;B",
    "证型选项": "A:脾胃虚弱;B:脾胃湿热;C:寒湿困脾;...",
    "病性": "虚证",
    "病性答案": "A",
    "病性选项": "A:虚证;B:实证;C:虚实夹杂;...",
    "病位": "脾胃",
    "病位答案": "A;B",
    "病位选项": "A:脾;B:胃;C:肝;D:肾;...",
    "治则治法": "健脾益气",
    "治则治法答案": "A;B",
    "治则治法选项": "A:健脾益气;B:清热化湿;C:温中散寒;...",
    "病因": "饮食不节,劳倦过度",
    "病机": "脾胃虚弱,运化失常",
    "治疗方法": "中药汤剂,针灸",
    "注意事项": "饮食调护,情志调节"
  },
  "disease_cn": "脾胃病",
  "disease_en": "Spleen and Stomach Disease",
  "exam_class": "中西医辩证分型"
}
```

---

## 🔧 配置

编辑 `ssdf_bench/configs/benchmark_config.example.json`：

```json
{
  "data_path": "evaluation_data/sdt_data.jsonl",
  "llm_judge_api_host": "127.0.0.1",
  "llm_judge_api_port": 8002,
  "llm_judge_model_name": "Qwen3-32B",
  "llm_judge_api_key": "",
  "max_retries": 3,
  "checkpoint_interval": 10,
  "random_seed": 42
}
```

---

## 📝 输出结果

```
results/your-model-run/
├── checkpoint.json                # 断点恢复文件
├── detailed_results.json          # 完整评估结果
├── evaluation_report.html         # 可视化 HTML 评估报告
└── logs_YYYYMMDD-HHMMSS.log       # 执行日志
```

---

## 🛠️ 依赖项

| 依赖 | 用途 |
|------------|---------|
| `torch` | 深度学习框架 |
| `transformers` | 模型加载与推理 |
| `openai` | LLM API 调用 |
| `fastapi` + `uvicorn` | Navigator API 服务 |
| `tqdm` | 训练与评估进度条 |
| `matplotlib` | 可视化报告 |
| `numpy` + `pandas` | 数据处理 |
| `json-repair` | 容错 JSON 解析 |
| `accelerate` + `deepspeed` | 分布式训练 |

---

## 📄 许可证

本项目基于 Apache License 2.0 许可证 — 详见 [LICENSE](LICENSE) 文件。

---

## 📚 引用

如您的研究中使用 DongYuan，请引用我们的论文：

```bibtex
@article{dongyuan2026dongyuan,
  title = {DongYuan: An LLM-Based Medical Expert System for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis},
  author = {Your Authors},
  journal = {Expert Systems with Applications},
  year = {2026},
  note = {Submitted for review}
}
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。无论是 Bug 修复、新功能建议还是文档改进，我们都非常感谢您的贡献。

---

## 🙏 致谢

- 感谢参与数据标注与症状标准化工作的中西医专家团队
- 感谢本研究所使用的开源基座模型（Qwen3-14B）的贡献者
- 感谢 SwanLab 平台提供的实验跟踪支持

---

## ⚠️ 免责声明

本项目仅供研究和教育用途。所有诊断结果仅供参考，不应作为临床医疗决策的依据。实际诊疗请咨询合格的医疗专业人员。

---

<p align="center">❤️ 用心支持中西医结合人工智能研究</p>
