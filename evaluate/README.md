# 评估模块

## 目录结构

```
evaluate/
├── conflict_detector.py    # 通用冲突检测器
├── validate_conflicts.py   # 冲突检测结果验证器
├── run_all.py              # 一键运行检测+验证
├── data/                   # 输入数据目录
│   ├── chunks.json         # chunk数据
│   └── error_data.json     # 标准答案数据
├── methods/                # 方法实现目录
│   ├── base.py             # BaseMemoryAPI 抽象基类
│   ├── openai/
│   │   └── chat.py         # OpenAI (对话历史记忆)
    ...
└── results/                # 结果输出目录
```

## 工作原理

冲突检测器的核心逻辑：

1. **逐个发送chunk**：每次只发送当前chunk内容给LLM/Agent
2. **要求记忆**：告诉LLM/Agent需要记住这些内容（调用 `store()`）
3. **自主检索**：让LLM/Agent自己检索之前记忆的内容（调用 `query()`）
4. **冲突检测**：判断当前内容是否与记忆中的历史内容有冲突

这种设计用于评估不同方法（如MemGPT、AMEM、LangMem等记忆增强方法）的记忆和检索能力。

## 使用方法

### 一键运行（推荐）

```bash
# 使用默认方法(openai) 运行检测+验证
python evaluate/run_all.py

# 指定方法
python evaluate/run_all.py -m amem
python evaluate/run_all.py -m memgpt
python evaluate/run_all.py -m langmem
```

### 单独运行冲突检测

```bash
# 使用默认方法(openai)
python evaluate/conflict_detector.py

# 指定方法
python evaluate/conflict_detector.py --method amem

# 指定输入输出文件
python evaluate/conflict_detector.py --method openai --chunks evaluate/data/chunks.json --output evaluate/results/custom_output.json

# 列出所有可用方法
python evaluate/conflict_detector.py --list-methods
```

### 单独运行结果验证

```bash
python evaluate/validate_conflicts.py --results evaluate/results/openai_conflicts.json
```

### 命令行参数

**conflict_detector.py:**

- `--method, -m`: 使用的方法名称 (默认: openai)
- `--chunks, -c`: chunks.json文件路径 (默认: evaluate/data/chunks.json)
- `--output, -o`: 输出文件路径 (默认: evaluate/results/{method}_conflicts.json)
- `--list-methods, -l`: 列出所有可用的方法

**run_all.py:**

- `--method, -m`: 使用的方法名称 (默认: openai)

## Memory API 接口规范

所有方法均继承自 `BaseMemoryAPI`（定义在 `methods/base.py`），需实现以下接口：

```python
from evaluate.methods.base import BaseMemoryAPI

class ChatAPI(BaseMemoryAPI):
    def store(self, content: str, time: str = None) -> bool:
        """储存内容到记忆系统"""
        pass

    def query(self, question: str, system_prompt: str = None, temperature: float = None) -> str:
        """查询记忆并生成回答"""
        pass

    def clear(self) -> None:
        """清空记忆"""
        pass

    def get_memory_count(self) -> int:
        """获取记忆条目数量（可选）"""
        return 0
```

每个 `chat.py` 还需提供模块级便捷函数，以便检测器动态加载：

```python
def get_chat_api() -> ChatAPI:
    """获取 ChatAPI 单例实例"""

def store(content: str, time: str = None) -> bool:
    """储存内容到记忆系统"""

def query(question: str, system_prompt: str = None, temperature: float = None) -> str:
    """查询记忆并生成回答"""
```

## 已实现方法

| 方法 | 记忆机制 | 底层依赖 | 说明 |
|------|---------|---------|------|
| **openai** | 对话历史列表 | OpenAI API (OpenRouter) | 将所有对话存为 message list，每次查询带上完整历史 |
| **amem** | AgenticMemory | A-Mem + 本地 Embedding | 使用 `add_note` / `find_related_memories_raw` 进行语义检索 |
| **memgpt** | Letta Agent | Letta Cloud API | 每个 session 创建独立 Agent，自动管理记忆 |
| **langmem** | InMemoryStore + Memory Tools | langgraph + langmem + 本地 Embedding | 使用 `create_react_agent` 搭配 langmem 的 memory tools 进行记忆管理 |

## 添加新方法

1. 在 `methods/` 下创建新的子文件夹（如 `methods/mymethod/`）
2. 在子文件夹中创建 `chat.py` 文件
3. 定义 `ChatAPI` 类，继承 `BaseMemoryAPI`，实现 `store` / `query` / `clear` 方法
4. 提供模块级 `store()` / `query()` / `get_chat_api()` 函数
5. 运行时使用 `--method mymethod` 指定新方法

## 输出格式

### 冲突检测结果 (`{method}_conflicts.json`)

```json
[
    {
        "current_chunk_id": 5,
        "conflict_type": 2,
        "old_sentence": "从记忆中检索到的原句",
        "new_sentence": "当前输入中的冲突句子",
        "description": "冲突描述",
        "method": "openai"
    }
]
```

### 统计信息 (`{method}_conflicts_stats.json`)

```json
{
    "method": "openai",
    "total_conflicts": 5,
    "type_distribution": {
        "1": 2,
        "2": 2,
        "3": 1
    }
}
```

### 验证结果 (`{method}_conflicts_validation.json`)

包含每个检测冲突与标准答案的匹配情况，以及 Precision / Recall / F1 / F0.5 / F2 等指标。

## 冲突类型

- **类型1**: 数值冲突 — 相同指标的数值不一致
- **类型2**: 语义冲突 — 相同概念的描述出现矛盾
- **类型3**: 逻辑冲突 — 前后逻辑关系矛盾

## 配置

所有 API 密钥和模型参数在项目根目录的 `config/config.cfg` 中配置：

- `[openai]`: OpenRouter API Key / Base URL / 模型名
- `[memGPT]`: Letta Cloud API Key
- `[AMEM]`: A-Mem 相关参数（如 `retrieve_k`）

