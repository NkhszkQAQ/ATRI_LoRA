# 运行配置

`runtime.example.json` 由 `atri-memory-rag --config` 和旧命令 `python tools/serve_memory_rag.py --config` 共同读取。它只引用模型、资料和私有文件，不包含用户 token 或签名密钥正文。

填写私有副本后启动；模型身份校验通过前不会将指定权重用于生成。示例中的 1.1-R SHA 是权重身份，不表示所有模型功能通过验收。运行代码版本、权重版本和检索资料版本分别记录。

新入口维持 `memory-rag-v1.2-dev` 隔离配置；不替换旧端口或旧数据库。`planner_schema` 与补充 dense 检索仍需显式启用。4B/DT 不适用此 8B 生命周期配置，继续使用原兼容入口。

完整入口和影响见 `docs/INTEGRATION.md`。
