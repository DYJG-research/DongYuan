# DongYuan（东垣）：以基于知识的双模型协同推理机制为核心的中西医结合脾胃病诊疗框架

<p align="center">
  <img src="https://img.shields.io/badge/框架-DongYuan-blue.svg" alt="DongYuan">
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/许可证-Apache%202.0-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/专科-中西医结合-brightgreen.svg" alt="Integrative Medicine">
</p>

<p align="center">
  <b>DongYuan（东垣）</b>是一个面向中西医结合（ICWM）脾胃病诊疗的框架。框架以基于知识的双模型协同推理机制（KG-DMCRM）为核心，将领域知识分别注入两个异构模型——核心诊疗大模型（<i>SSDF-Core</i>）与轻量级问诊导航模型（<i>SSDF-Navigator</i>）——并协同二者完成多轮主动问诊。框架在<i>SSDF-Bench</i>上进行评估——这是一个过程-结果耦合的自建基准，配备经过验证的 LLM-as-a-judge 协议。
</p>

---

## 📋 项目简介

大语言模型（LLM）在医学诊断任务上已展现出性能提升，但系统性地将领域知识注入模型、并支撑中西医结合诊疗所需的多轮主动问诊，仍然困难重重。这一挑战在中西医结合脾胃病诊疗中尤为突出——中医辨证分型与西医客观证据需要在交互式诊断流程中被联合推理。

**DongYuan** 是一个以基于知识的双模型协同推理机制（KG-DMCRM）为核心的中西医结合脾胃病诊疗框架。KG-DMCRM 将领域知识分别注入两个异构模型并协同二者完成多轮主动问诊：核心诊疗大模型（*SSDF-Core*）与轻量级问诊导航模型（*SSDF-Navigator*）。框架进一步引入过程-结果耦合的自建评估基准（*SSDF-Bench*），配备经过验证的 LLM-as-a-judge 协议。

实验表明，**SSDF-Core** 在 **SSDF-Bench** 上优于 12 个主流基线模型，且将 **SSDF-Navigator** 与 **SSDF-Core** 协同可在多轮主动问诊中提升关键症状召回率与问诊效率。

### 核心特性

- **基于知识的双模型协同推理机制（KG-DMCRM）**：将领域知识分别注入核心诊疗大模型与问诊导航模型，并通过基于候选的协同机制协调二者完成多轮主动问诊
- **核心诊疗大模型（SSDF-Core）**：基于三个专家标注数据集经 SFT + DPO 两阶段训练，联合推理中医辨证分型与西医客观证据
- **问诊导航模型（SSDF-Navigator）**：基于本体状态抽象（83 个标准证型）与混合模仿-强化学习，提供可控、可优化的主动问诊策略
- **过程-结果耦合评估基准（SSDF-Bench）**：配备经过验证的 LLM-as-a-judge 协议，同时评估诊断结果与推理过程

---

## 🏗️ 系统架构

```
DongYuan/
├── kg_dmcrm/                              # 基于知识的双模型协同推理机制
│   ├── coordination/                      # KG-DMCRM 双模型协同
│   │   └── llm_navigator_coordination.py  #   SSDF-Core 与 SSDF-Navigator 的协同编排
│   ├── ssdf_core/                         # SSDF-Core：核心诊疗模型
│   │   ├── run_ssdf_core_sft_training.sh  #   监督微调（SFT）脚本
│   │   └── run_ssdf_core_dpo_training.sh  #   直接偏好优化（DPO）脚本
│   └── ssdf_navigator/                    # SSDF-Navigator：问诊导航模型
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
│   ├── mpc_evaluation/                  # 多轮主动问诊（MPC）评估
│   │   ├── run_mpc_collaborative_evaluation.py # 协同问诊评估
│   │   ├── run_mpc_pure_llm_evaluation.py      # 纯 LLM 问诊评估
│   │   ├── llm_patient_simulator.py            # 患者模拟器
│   │   ├── navigator_api_service.py            # Navigator API 服务
│   │   └── evaluate_mpc.py                     # MPC 结果分析
│   ├── evaluation_prompts/              # 评估提示词模板
│   ├── configs/                         # 配置文件
│   └── scripts/                         # 部署与评估脚本
└── data_foundation/                     # 专家标注数据集
    ├── evaluation_data/                 # 评估数据集
    │   ├── sdt_data.jsonl               # SSDF-Syndrome：辨证论治案例
    │   ├── mpc_data.json                # SSDF-PD：多轮问诊评估数据
    │   └── other_task_data.json         # 选择题数据（中医、西医、伦理、安全）
    └── train_data/                      # 训练数据集
        ├── sft_data/                    # SFT 数据（COT、多轮对话、考试增强）
        └── dpo_data/                    # DPO 偏好数据
```

---

## 🔬 方法论

### KG-DMCRM：基于知识的双模型协同推理机制

DongYuan 的核心是 KG-DMCRM，它系统性地将领域知识注入两个异构模型——核心诊疗大模型（*SSDF-Core*）与问诊导航模型（*SSDF-Navigator*）——并通过基于候选的协同机制协调二者完成多轮主动问诊。对每个模型而言，领域知识都通过「知识载体」与「注入方式」两个环节完成注入：

#### SSDF-Core：核心诊疗模型

**专家标注数据集（知识载体）。** SSDF-Core 的知识注入建立在三个自建、专家标注的数据集之上。这些数据集显式编码专家临床推理而非简单的输入-输出对，均由中西医结合医生在双盲标注与第三方仲裁的流程下构建：

| 数据集 | 描述 |
|---------|------|
| **SSDF-Syndrome** | 辨证论治推理链（1,345 条），包含完整患者病历、舌诊脉诊、检查检验结果及金标准证型标签 |
| **SSDF-Dialogue** | 多轮医患问诊对话及结构化推理模板（1,080 条），映射到 83 个标准化证型 |
| **SSDF-PD** | 专家决策偏好数据（12,800 对 chosen-rejected），编码专家对主动问诊与辨证论治的决策偏好 |

**两阶段训练（注入方式）。** 为了将数据集承载的领域知识深入注入模型，SSDF-Core 在 Qwen3-14B 基座上采用 SFT + DPO 两阶段训练：先通过监督微调（SFT）学习专家诊断推理，再通过直接偏好优化（DPO）对齐专家决策偏好，使其能够联合推理中医辨证分型与西医客观证据。

#### SSDF-Navigator：问诊导航模型

**基于本体的状态抽象（知识载体）。** 为了提供可控、可优化的问诊策略，SSDF-Navigator 将原始自然语言问诊历史映射到结构化状态空间：在中医临床术语国家标准与中西医结合医生指导下，将医生发起的问诊问题映射为 83 个标准脾胃病证型，从而把原始问诊历史转换为证型状态序列。

**混合模仿-强化学习（注入方式）。** SSDF-Navigator 是一个轻量级 Transformer 编码器-分类器策略网络，通过行为克隆（BC）与信息增益奖励驱动的离线强化学习（RL）混合训练，学习专家的递进式问诊策略：

- **问题定义**——将多轮问诊建模为序贯决策问题：
  - **状态（State）**：最近 $L$ 个历史标准化证型构成的状态窗口 $state_t = [s_{t-L+1}, ..., s_t]$
  - **动作（Action）**：专家下一步询问的标准化证型 $a_t = s_{t+1}$
  - **策略（Policy）**：学习 $\pi(a_t | state_t)$ 以模拟专家递进式问诊逻辑
- **模型架构**——SSDF-Navigator 采用 Transformer 编码器-分类器设计（嵌入层 + 正弦位置编码 → Transformer 编码器 → 掩码平均池化 → 83 维证型分类头）
- **信息增益奖励函数**：$R_{info}(a) = 1 - \tilde{H}(\mathcal{S} | a)$，其中 $\tilde{H}(\mathcal{S} | a)$ 是查询证型 $a$ 后的归一化条件熵，并引入重复因子 $\lambda(a, x, H_t, \mathcal{M}_t)$
- **Reward-Weighted Behavior Cloning（RWBC）**（`--fusion_mode mul`）：$\mathcal{L}_{RWBC}(\theta) = -\frac{1}{B} \sum r_t \cdot \log \pi_{\theta}(a_t^* | X_t)$——以奖励作为样本权重
- **Reward-Added Behavior Cloning（RABC）**（`--fusion_mode add`）：$\mathcal{L}_{RABC}(\theta) = \beta_1 \mathcal{L}_{BC} + \beta_2 \mathcal{L}_{PO} + \mathcal{L}_{Ent}$——结合监督损失、奖励驱动的策略优化与熵正则化

#### 双模型协同：基于候选的协同机制

两个模型在闭环诊断流程中通过基于候选的协同机制协作：

1. SSDF-Core 基于当前患者信息生成候选问诊问题
2. 将问题映射到标准化证型空间
3. SSDF-Navigator 评估当前证型状态，输出 top-K 证型推荐及其概率分布
4. SSDF-Core 从候选集（Navigator 的 top-K 推荐 + SSDF-Core 自身候选）中选择下一步问诊
5. 新的患者回答更新证型状态，循环继续直至获取足够信息进行诊断

### SSDF-Bench：过程-结果耦合评估

SSDF-Bench 提供综合性评估框架，同时评估诊断结果与推理过程。它包含六个任务，其中两个是中西医结合脾胃病诊疗的核心临床任务（SDT 与 MPC），四个是评估疾病通用医学知识、安全与伦理的常规选择题任务（TCM-BC、WM-BC、MS、ME）。

**辨证论治（SDT）维度：**

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

**附加评估任务：**

| 任务 | 描述 | 权重 |
|----------|-------------|--------|
| TCM-BC（中医基础知识） | 中医基础知识 | 20% |
| WM-BC（西医基础知识） | 西医知识 | 20% |
| MS（医学安全） | 临床边界合规与安全价值对齐 | 10% |
| ME（医学伦理） | 医学伦理理解与判断 | 10% |

**多轮主动问诊（MPC）评估：** 使用 LLM 模拟患者，评估完整多轮问诊中的关键症状召回率、问诊效率与诊断准确率。

**评分体系：** SDT（40%）+ TCM-BC（20%）+ WM-BC（20%）+ MS（10%）+ ME（10%）。选择题评分采用 Sp 指标：$Sp = |A \cap B| / (|A| + |\bar{A} \cap B|)$。

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

### 训练 SSDF-Navigator（问诊导航模型）

```bash
cd kg_dmcrm/ssdf_navigator

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

### 训练 SSDF-Core（核心诊疗模型）

**阶段一——监督微调（SFT）：**
```bash
cd kg_dmcrm/ssdf_core
bash run_ssdf_core_sft_training.sh
```

**阶段二——直接偏好优化（DPO）：**
```bash
bash run_ssdf_core_dpo_training.sh
```

### 运行 KG-DMCRM 协同问诊

```bash
cd ssdf_bench/mpc_evaluation

# 纯 LLM 基线（仅 SSDF-Core）
python run_mpc_pure_llm_evaluation.py \
  --core_model "your-model" \
  --core_model_base_url "http://localhost:8000/v1"

# KG-DMCRM 协同评估（SSDF-Core + SSDF-Navigator）
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

## 📥 数据下载

DongYuan 自建的核心数据集（SSDF-Syndrome、SSDF-Dialogue、SSDF-PD）已直接包含在仓库的 `data_foundation/` 目录下。补充训练数据（通用指令、医学知识、伦理、安全等）因体积较大未包含在仓库中，请通过以下链接单独下载：

- **补充训练数据**（`data_foundation/train_data/sft_data/other_data/`）：[https://pan.quark.cn/s/9633dc0ed2da](https://pan.quark.cn/s/9633dc0ed2da)

下载后将文件解压到 `data_foundation/train_data/sft_data/other_data/` 目录即可复现完整训练环境。

---

## 📄 许可证

本项目基于 Apache License 2.0 许可证 — 详见 [LICENSE](LICENSE) 文件。

---

## 📚 引用

如您的研究中使用 DongYuan，请引用我们的论文：

```bibtex
@article{dongyuan2026framework,
  title = {DongYuan: A Framework for Integrative Chinese and Western Medicine Spleen-Stomach Disorders Diagnosis with a Knowledge-Grounded Dual-Model Collaborative Reasoning Mechanism},
  author = {Your Authors},
  journal = {Journal of Biomedical Informatics},
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
