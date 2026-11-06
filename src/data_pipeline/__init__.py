"""Dreamy 数据处理流水线：清洗 -> 配比混合 -> 打包。

模块职责：
    schema    统一数据格式与读取适配
    clean     规范化、质量过滤、去重、脱敏
    mix       按配比采样
    report    健康度报告
    tokenizer 训练 SentencePiece 并打包 token 流
"""
