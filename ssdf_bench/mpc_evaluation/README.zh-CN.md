# SSDF  Consultation Navigator 评估系统
> 用于评估 SSDF Transformer 导航模型在脾胃病诊断中多轮问诊能力的综合评估框架

## 项目简介

`exam` 模块提供了一个全面的评估框架，用于测试 SSDF（脾胃病基础模型）问诊导航模型在脾胃病领域进行多轮中医问诊的性能。

该评估系统通过以下方式模拟真实医学问诊场景：
1. 使用 **基于 LLM 的患者模拟器** 根据真实病历生成患者回答
2. 运行**多轮对话**，模拟医生模型与患者之间的问诊过程
3. 评估医生模型的问诊能力

## 项目结构

```
exam/
├── Multiround_exam.py          # 生成测试的对话结果
├── LLM_as_Patient.py           # 基于 LLM 的患者模拟器
├── LLMAndNavigator.py          # 大模型和navigator小模型结合脚本
├── Eval.py                     # 评估脚本，对对话结果进行评估
├── exam_datas/                 # 评估数据集
│   └── exam_datas.json         # 主测试用例（病历记录）
├── datas/                      # 评估结果
│   ├── eval_results/           # 评估输出结果
│   └── exam_chat_records_*.json # 问诊记录
└── logs/                      # 执行日志
```

## 核心组件

### 1. 患者模拟器 (`LLM_as_Patient.py`)
基于 LLM 的患者模拟器，根据以下信息生成真实的患者回答：
- 病历信息（主诉、症状、病史）
- 预定义的回答规则，确保一致性和真实性

**功能特点：**
- 支持两种模式：默认模式（基于主诉）和病历模式
- 保持对话一致性
- 生成口语化的中文回答，模拟真实患者

### 2. 评估引擎 (`Multiround_exam.py`)
主导评估脚本，负责协调整个问诊过程：

**核心功能：**
- `SSDFConsultationClient`：带导航器协作的医生模型客户端
- 多轮问诊循环（最多 30 轮）
- 问诊结束时自动生成诊断结论
- 完整的对话日志记录

### 3. 融合方法
支持导航器与医生 LLM 之间的多种协作策略：
- `llm_choose`：导航器推荐症状，LLM 决定是否采纳
- `PureLLM`：不使用导航器，直接使用 LLM 进行问诊

## 使用方法

### 前置条件

1. **运行中的服务：**
   - SSDF-core 模型服务（医生模型）
   - 导航器服务（症状预测）
   - 患者模拟器模型服务（如 Qwen3-32B）

2. **必需文件：**
   - 症状词汇表文件（`symptoms.csv`）
   - 问题标准化文件（`question_cleaned.csv`）
   - 评估数据集（`exam_datas.json`）

### 运行评估

**标准评估（带导航器）：**
```bash
python Multiround_exam.py \
    --fusion_method llm_choose \
    --file-diagnose-records ./exam_datas/clearned_3_100.json \
    --file-save-path ./datas/exam_chat_records_llm_choose_20260305.json \
    --ssdf-core-model-base-url http://localhost:8008/v1 \
    --navigator-url http://localhost:8999/predict \
    --patient-base-url http://localhost:11118/v1 \
    --patient-model-name Qwen3-32B
```

**纯 LLM 评估（不带导航器）：**
```bash
python Multiround_examPureLLM.py \
    --fusion_method PureLLM \
    --file-diagnose-records ./exam_datas/clearned_3_100.json \
    --file-save-path ./datas/exam_chat_records_pureLLM.json \
    --ssdf-core-model-base-url http://localhost:8008/v1 \
    --patient-base-url http://localhost:11118/v1
```

### 关键参数说明

| 参数 | 说明 | 默认值 |
|----------|-------------|---------|
| `--fusion_method` | 协作方式 | `llm_choose` |
| `--file-diagnose-records` | 测试数据集路径 | 必填 |
| `--file-save-path` | 输出保存路径 | 必填 |
| `--ssdf-core-model-base-url` | 医生模型 API | `http://localhost:8008/v1` |
| `--navigator-url` | 导航器服务 URL | `http://localhost:8999/predict` |
| `--patient-base-url` | 患者模拟器 API | `http://localhost:11118/v1` |
| `--history-length` | 症状历史窗口长度 | `5` |
| `--navigator-topk` | 导航器返回的 Top-K 症状数 | `5` |
| `--low-threshold` | 导航器概率阈值 | `0.1` |

## 评估输出

评估过程会生成以下结果：

### 1. 问诊记录（`exam_chat_records_*.json`）
- 完整的多轮对话历史
- 医生提问和患者回答
- 最终诊断结论

### 2. 评估日志（`logs/`）
- 详细的执行日志
- 带时间戳的对话轮次
- 错误追踪

### 3. 统计结果（`datas/statistics_results.ipynb`）
- 对话长度分析
- 症状覆盖率指标
- 诊断准确率评估

## 与主项目集成

该评估模块是 **SSDF-navigator** 项目的一部分：

```
SSDF-navigator/
├── exam/                      # 评估模块（当前文件夹）
├── train.py                   # 模型训练脚本
├── evaluate.py                # 导航器评估
└── [其他模块]
```

有关导航器模型架构和训练的详细信息，请参阅主项目 README。

## 技术细节

### 问诊流程

1. **初始化**：加载患者病历，初始化医生客户端和患者模拟器
2. **主诉**：从病历中获取患者的主要症状开始问诊
3. **多轮循环**：
   - 医生生成问诊问题（带/不带导航器指导）
   - 患者模拟器根据病历回答
   - 检查问诊是否应该结束
4. **诊断**：生成中医证型和西医诊断
5. **记录**：保存完整的问诊记录

### 导航器协作机制

导航器通过以下方式辅助医生模型：
1. 根据对话历史预测下一个最相关的 K 个症状
2. 为每个症状提供概率分数
3. 医生模型决定是否采纳导航器的推荐

## 许可证

本项目是 SSDF-navigator 研究项目的一部分。许可证和使用条款请参阅主项目文档。

## 致谢

- 中医专家提供的问诊逻辑指导
- 病历数据提供方
- 开源 LLM 项目（Qwen 等）
