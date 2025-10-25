# 🧩 文本冲突数据集

## 📘 数据格式

每条样本为一个 JSON 对象，数据集整体为一个 JSON 数组。

```json
[
    {
        "category": 3,
        "original_statement": {
            "statement": "",
            "chunk": 1,
            "position": 3
        },
        "conflicting_statement": {
            "statement": "",
            "chunk": 2,
            "position": 7
        },
        "chunk_distance": 1,
        "similarity": 0.3,
        "contradiction_level": 1,
        "source": "auto"
    }
]
```

---

## 🧱 字段说明

| 字段 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `category` | `int` | `3` | 冲突类型编号（见下方分类表）。 |
| `original_statement` | `object` | `{ "statement": "术后需要休息十天", "chunk": 1, "position": 3 }` | 原始句或段落。 |
| ┗ `statement` | `string` | `"术后需要休息十天"` | 原始信息文本。 |
| ┗ `chunk` | `int` | `1` | 所在文本块编号。 |
| ┗ `position` | `int` | `3` | 文本位置（句号编号、字符索引等）。 |
| `conflicting_statement` | `object` | `{ "statement": "术后需要立即运动", "chunk": 2, "position": 7 }` | 与原文矛盾的句子。 |
| `chunk_distance` | `int` | `2` | 两个冲突句所在 chunk 的距离。 |
| `similarity` | `float` | `0.3` | 语义相似度（0~1）。 |
| `contradiction_level` | `int` | `0` | 冲突层级：`0`=隐性；`1`=显性。 |
| `source` | `string` | `"auto"` | 样本来源：`"auto"`（自动生成）或 `"manual"`（人工标注）。 |

## 🧠 contradiction_level 说明

| 值 | 类型 | 定义 | 示例 |
|----|------|------|------|
| **0** | **隐性冲突（Implicit）** | 语义或逻辑层矛盾，需要常识或上下文理解。 | “术后应避免运动” ↔ “医生建议术后进行康复训练”。 |
| **1** | **显性冲突（Explicit）** | 字面语义直接对立，无需推理。 | “禁止吸烟” ↔ “可以吸烟”。 |

---

## 📚 冲突类型分类表（category）

| ID | 冲突类型 | 定义 | 示例 |
|----|------------|------|------|
| **1** | **数值冲突（Numerical Conflict）** | 对同一事实、数量、时间或范围的描述不一致。 | “血压在60–100之间为正常” ↔ “血压在80–120之间为正常”；“每天服药两次” ↔ “每天服药一次”。 |
| **2** | **语义冲突（Semantic Conflict）** | 对同一行为或建议表达相反的语义。 | “术后需要休息十天” ↔ “术后需要立即运动”；“禁止饮酒” ↔ “建议每天少量饮酒”。 |
| **3** | **逻辑冲突（Logical Conflict）** | 因果、条件或推理方向互相矛盾。 | “如果摄入过多糖，可能得糖尿病” ↔ “得了糖尿病一定是因为摄入糖过多”；“因为吃药他康复了” ↔ “因为吃药他生病了”。 |



---

## 🧪 示例样本

```json
[
    {
        "category": 2,
        "original_statement": {
            "statement": "术后需要休息十天。",
            "chunk": 1,
            "position": 3
        },
        "conflicting_statement": {
            "statement": "术后需要立即运动。",
            "chunk": 2,
            "position": 7
        },
        "chunk_distance": 2,
        "similarity": 0.38,
        "contradiction_level": 0,
        "source": "auto"
    }
]
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install openai sentence-transformers numpy

# 配置API密钥
# 在 ../config/config.cfg 中设置 ali_api_key
```

### 2. 数据集构建

#### 2.1 准备源文档
将长文本文档放置在 `dataset/data.txt`

#### 2.2 运行自动构建
```bash
cd error_detect
python data_construct.py
```

**生成文件:**
- `dataset/chunks.json` - 文档分块结果(包含原始文本和插入冲突后的文本)
- `dataset/error_data1.json` - 冲突数据集
- `dataset/error_data1_stats.json` - 统计信息
- `logs/data_construct_*.log` - 详细运行日志

#### 2.3 更新位置信息(可选)
```bash
python update_positions.py
```

自动定位陈述在chunk中的精确句子位置,支持4级匹配策略(精确→包含→规范化→模糊)。

---

## 📦 项目结构

```
error_detect/
├── data_construct.py       # 主构建脚本(LLM自动生成冲突数据集)
├── prompts.py              # 提示词模板库(提取陈述+生成冲突)
├── update_positions.py     # 位置更新工具(精确定位句子位置)
├── README.md               # 项目文档
├── dataset/
│   ├── data.txt           # 源文档(长文本)
│   ├── chunks.json        # 分块结果(original_text + processed_text)
│   ├── error_data1.json   # 冲突数据集
│   └── error_data1_stats.json  # 统计信息
└── logs/
    └── data_construct_*.log    # 运行日志(含LLM对话详情)
```

---

## 🔧 核心功能

### 1. 文档分块 (`load_and_chunk_document`)
- 按句子边界分割,不切断句子
- 默认分成10个chunk
- 保留原始文本(`original_text`)和处理文本(`processed_text`)

### 2. 陈述提取 (`extract_key_statements`)
- 使用Qwen-Max从每个chunk提取关键陈述
- 优先提取包含数值、频率、时间的陈述
- 确保逐字逐句从原文提取(temperature=0.1)

### 3. 冲突生成 (`generate_conflicting_statement`)
- 三种冲突类型,各有专属提示词:
  - **类型1**: 数值冲突(改变数值/范围/频率/时间)
  - **类型2**: 语义冲突(表达相反建议或行为)
  - **类型3**: 逻辑冲突(改变因果/条件关系)
- 使用同义替换和句式重组,避免简单否定
- 保持专业性和自然性

### 4. 相似度计算 (`calculate_cosine_similarity`)
- 使用sentence-transformers计算实际余弦相似度
- 模型: `all-MiniLM-L6-v2`
- 确保冲突陈述与原陈述有足够差异

### 5. 位置定位 (`find_sentence_position`)
**4级匹配策略:**
1. **精确匹配**: 字符串完全相等
2. **包含匹配**: 陈述与句子互相包含
3. **规范化匹配**: 去除标点和空格后比对
4. **模糊匹配**: 基于字符重叠率(阈值80%)

### 6. 日志记录
所有运行流程、LLM提示词、返回内容均记录到日志文件:
- 系统/用户提示词完整内容
- API请求参数(model, temperature, max_tokens)
- LLM返回的原始响应
- 解析结果和统计信息

---

## ⚙️ 配置参数

在 `data_construct.py` 的 `main()` 函数中修改:

```python
# 文件路径
input_file = "dataset/data.txt"
chunks_file = "dataset/chunks.json"
output_file = "dataset/error_data1.json"

# 构建参数
num_chunks = 10              # 分块数量
num_conflicts = 1            # 生成冲突对数量
min_chunk_distance = 3       # 最小块距离(冲突陈述必须在原陈述之后)
```

---

## 📊 数据集统计

运行后自动生成 `error_data1_stats.json`:

```json
{
    "total_conflicts": 10,
    "category_distribution": {
        "1": 3,
        "2": 4,
        "3": 3
    },
    "contradiction_level_distribution": {
        "0": 6,
        "1": 4
    },
    "avg_chunk_distance": 4.2,
    "avg_similarity": 0.35,
    "source_distribution": {
        "auto": 10
    }
}
```

---

## 🎯 设计原则

### 1. 两阶段流程
- **阶段1**: 从原始文本提取关键陈述
- **阶段2**: 基于陈述生成冲突,插入到后续chunk

### 2. 位置约束
- 冲突陈述必须出现在原陈述**之后**的chunk
- 默认最小距离3个chunk(可配置)
- 如无法满足,则选择任意后续chunk

### 3. 质量保证
- 提取时使用低温度(0.1)确保准确
- 生成时使用中温度(0.7)保持创意
- 计算实际余弦相似度
- 4级匹配策略确保位置准确

### 4. 可追溯性
- 详细日志记录所有LLM交互
- 保留原始文本和处理文本
- 记录每个冲突对的来源和参数

---
## 📄 许可证

MIT License

---

## 👥 贡献

欢迎提交Issue和Pull Request改进数据集质量!

