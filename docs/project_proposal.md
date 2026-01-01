# Hello LLM Fine-Tuning | 大模型应用开发实战二：微调技术全栈指南

## 项目方案设计

> 参考 all-in-rag 项目的组织结构，设计一个面向大模型微调的开源教程项目

---

## 一、项目定位

**项目名称**: Hello LLM Fine-Tuning  
**副标题**: 大模型微调技术全栈指南 - 从理论到实践，从基础到进阶，构建你的微调技术体系

**目标受众**:
- 具备 Python 编程基础，对大模型微调感兴趣的开发者
- 希望系统学习微调技术的 AI 工程师
- 想要定制化大模型能力的产品开发者
- 对微调技术有学习需求的研究人员

**前置要求**:
- 掌握 Python 基础语法和 PyTorch 基本使用
- 了解 Transformer 架构基础概念
- 具备基础的 Linux 命令行操作能力
- 能够简单使用 Docker（推荐但非必需）

---

## 二、目录结构设计

```
all-in-finetuning/
├── docs/                    # 教程文档
│   ├── index.html          # docsify 首页
│   ├── _sidebar.md         # 侧边栏导航
│   ├── README.md           # 文档首页
│   ├── chapter1/           # 第一章：微调基础入门
│   ├── chapter2/           # 第二章：数据准备
│   ├── chapter3/           # 第三章：微调方法详解
│   ├── chapter4/           # 第四章：训练优化技术
│   ├── chapter5/           # 第五章：模型评估与部署
│   ├── chapter6/           # 第六章：高级微调技术
│   ├── chapter7/           # 第七章：项目实战一（基础篇）
│   ├── chapter8/           # 第八章：项目实战二（进阶篇）
│   └── en/                 # 英文版本
├── Extra-chapter/           # 扩展章节
│   └── README.md
├── .github/                 # GitHub 配置
│   └── workflows/
├── .gitignore
├── README.md               # 项目说明（中文）
├── README_en.md            # 项目说明（英文）
└── LICENSE
```

---

## 三、已完成内容大纲

### 第零章 导读
- [x] [AI大模型四阶通关指南](chapter0/00_overview.md) - 从提示词到预训练的技术路线图

### 第一章 微调前置知识
- [x] [提示工程实战指南](chapter1/01_prompt_engineering.md) - 零训练成本解锁大模型潜力
- [x] [从零构建AI智能体](chapter1/02_ai_agent.md) - ReAct范式与LangChain实战

### 第二章 理论基础
- [x] [语言模型演进全解析](chapter2/01_lm_evolution.md) - 从统计到GPT/BERT的核心突破
- [x] [大模型预训练技术](chapter2/02_pretraining.md) - 从通识教育到千亿参数炼成

### 第三章 微调入门
- [x] [大模型微调实战指南](chapter3/01_finetuning_overview.md) - 从理论到高效落地的核心路径
- [x] [Hugging Face Transformers入门](chapter3/02_transformers_start.md) - 大模型微调入门实战

### 第四章 PEFT参数高效微调
- [x] [PEFT实战指南](chapter4/01_peft_intro.md) - 用1%参数高效微调大模型
- [x] [PEFT主流技术全解析](chapter4/02_peft_categories.md) - 四大核心类别与实战选型
- [x] [Soft Prompt实战](chapter4/03_soft_prompt.md) - Prefix Tuning vs Prompt Tuning
- [x] [P-Tuning演进全解析](chapter4/04_p_tuning.md) - 从V1到V2的核心技术
- [x] [Adapter微调](chapter4/05_adapter.md) - 用3.6%参数撬动大模型100%性能
- [x] [LoRA低秩适配微调](chapter4/06_lora.md) - 百万参数撬动十亿大模型

### 第五章 模型部署
- [x] [vLLM推理引擎](chapter5/01_vllm.md) - 核心加速机制与组件原理
- [x] [Triton部署实战](chapter5/02_triton.md) - 从设计思想到生产落地

### 待扩展内容（规划中）
- [ ] 数据准备与处理
- [ ] 训练优化技术（显存优化、分布式训练）
- [ ] 人类偏好对齐（RLHF、DPO）
- [ ] 项目实战

---

## 四、技术栈选型

| 类别 | 工具/框架 | 说明 |
|------|----------|------|
| 基础框架 | PyTorch | 深度学习基础框架 |
| 模型库 | Transformers | Hugging Face 模型库 |
| 微调框架 | PEFT | 参数高效微调库 |
| 训练框架 | TRL | 强化学习训练库 |
| 分布式 | DeepSpeed / Accelerate | 分布式训练优化 |
| 数据处理 | Datasets | 数据集处理库 |
| 部署 | vLLM / TGI | 模型推理部署 |
| 评估 | lm-evaluation-harness | 模型评估工具 |

---

## 五、项目亮点设计

1. **体系化学习路径** - 从基础概念到高级应用，构建完整的微调技术学习体系
2. **理论与实践并重** - 每个章节都包含理论讲解和代码实践
3. **多种微调方法覆盖** - 全参数微调、LoRA、QLoRA、DPO 等主流方法
4. **工程化导向** - 注重显存优化、分布式训练、模型部署等工程实践
5. **丰富的实战项目** - 提供从基础到进阶的完整实战项目

---

## 六、实施计划

### 阶段一：基础框架搭建
- [ ] 创建项目目录结构
- [ ] 配置 docsify 文档系统
- [ ] 编写 README 和基础说明文档

### 阶段二：核心内容编写
- [ ] 第一章：微调基础入门
- [ ] 第二章：数据准备
- [ ] 第三章：微调方法详解
- [ ] 第四章：训练优化技术

### 阶段三：进阶内容编写
- [ ] 第五章：人类偏好对齐
- [ ] 第六章：模型评估与部署
- [ ] 第七章：高级微调技术

### 阶段四：实战项目开发
- [ ] 第八章：项目实战一
- [ ] 第九章：项目实战二

---

## 七、参考资源

- [Hugging Face PEFT 文档](https://huggingface.co/docs/peft)
- [Hugging Face TRL 文档](https://huggingface.co/docs/trl)
- [DeepSpeed 文档](https://www.deepspeed.ai/)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)

---

*方案版本: v1.0*  
*创建日期: 2026-01-02*
