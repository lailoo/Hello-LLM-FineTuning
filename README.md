<div align="center">
  <img src="./logo.png" alt="Hello LLM Fine-Tuning Logo" width="200">
  <h1>Hello LLM Fine-Tuning</h1>
  <h3>🔧 大模型微调技术全栈指南</h3>
  <p><em>从理论到实践，从基础到进阶，构建你的微调技术体系</em></p>
  
  <p>
    <a href="https://github.com/lailoo/Hello-LLM-FineTuning"><img src="https://img.shields.io/github/stars/lailoo/Hello-LLM-FineTuning?style=social" alt="GitHub stars"></a>
    <a href="https://github.com/lailoo/Hello-LLM-FineTuning/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" alt="License"></a>
  </p>
</div>

---

## 📖 项目简介

本项目是一个面向大模型应用开发者的微调（Fine-tuning）技术全栈教程，旨在通过体系化的学习路径和动手实践项目，帮助开发者掌握基于大语言模型的微调技术，构建定制化的智能应用。

**在线阅读**: [Hello LLM Fine-Tuning 文档](https://lailoo.github.io/Hello-LLM-FineTuning/)

## ✨ 项目亮点

- 🎯 **体系化学习路径** - 从基础概念到高级应用，构建完整的微调技术学习体系
- 📚 **理论与实践并重** - 每个章节都包含理论讲解和代码实践
- 🔧 **多种微调方法覆盖** - LoRA、QLoRA、Adapter、P-Tuning 等主流方法
- 🚀 **工程化导向** - 注重模型部署等工程实践

## 📚 内容大纲

| 章节 | 内容 |
|------|------|
| **第零章** | [AI大模型四阶通关指南](./docs/chapter0/00_AI大模型四阶通关指南.md) |
| **第一章** | 微调前置知识：提示工程、AI智能体 |
| **第二章** | 理论基础：语言模型演进、预训练技术 |
| **第三章** | 微调入门：实战指南、Transformers入门 |
| **第四章** | PEFT参数高效微调：LoRA、Adapter、P-Tuning 等 |
| **第五章** | 模型部署：vLLM、Triton |

## 🎯 目标受众

- 具备 Python 编程基础，对大模型微调感兴趣的开发者
- 希望系统学习微调技术的 AI 工程师
- 想要定制化大模型能力的产品开发者

## 📋 前置要求

- 掌握 Python 基础语法和 PyTorch 基本使用
- 了解 Transformer 架构基础概念
- 具备基础的 Linux 命令行操作能力

## 🚀 快速开始

### 本地预览文档

```bash
# 安装 docsify-cli
npm install -g docsify-cli

# 启动本地服务
docsify serve docs

# 访问 http://localhost:3000
```

## 📁 项目结构

```
Hello-LLM-FineTuning/
├── docs/                    # 教程文档
│   ├── index.html          # docsify 配置
│   ├── _sidebar.md         # 侧边栏导航
│   ├── README.md           # 文档首页
│   ├── chapter0/           # 第零章：导读
│   ├── chapter1/           # 第一章：微调前置知识
│   ├── chapter2/           # 第二章：理论基础
│   ├── chapter3/           # 第三章：微调入门
│   ├── chapter4/           # 第四章：PEFT参数高效微调
│   ├── chapter5/           # 第五章：模型部署
│   └── images/             # 图片资源
├── .github/                 # GitHub 配置
│   └── ISSUE_TEMPLATE/     # Issue 模板
├── logo.png                # 项目 Logo
├── LICENSE                 # MIT 许可证
└── README.md               # 项目说明（本文件）
```

## 🤝 参与贡献

欢迎提交 Issue 和 Pull Request！

- 🐛 [报告 Bug](.github/ISSUE_TEMPLATE/bug_report.yml)
- 💡 [功能建议](.github/ISSUE_TEMPLATE/feature_request.yml)
- 📚 [学习反馈](.github/ISSUE_TEMPLATE/study_feedback.yml)

## 📄 许可证

本项目采用 [CC BY-NC-SA 4.0 许可证](LICENSE)（知识共享署名-非商业性使用-相同方式共享 4.0 国际）