# Docker & Docker Compose 部署说明

本项目支持使用 Docker 和 docker-compose 一键部署，适用于模型微调、推理等场景。

---

## 1. 构建镜像

在项目根目录下执行：
```bash
docker-compose build
```

## 2. 启动容器

```bash
docker-compose up -d
```

## 3. 进入容器

```bash
docker exec -it fintuneqwen3 bash
```

## 4. 运行微调/推理脚本

容器内已安装所有依赖，可直接运行如：
```bash
python data_prepare/download_tcm_data.py
python model_finetune/finetune_sft.py
python model_download/load_local_model.py
```

## 5. 端口映射（如需API服务）

如需暴露端口（例如 FastAPI/Flask），请在 `docker-compose.yml` 的 `ports` 部分取消注释并设置端口。

---

## 目录与数据挂载
- 默认挂载当前目录到容器 `/app`，所有代码和数据可直接访问。
- `trainer_output` 文件夹不会被复制进镜像，避免大文件冗余。

---

## 依赖说明
- Python 3.10
- transformers
- datasets
- matplotlib
- torch
- peft

如需其他依赖，请在 Dockerfile 中补充。

---

## 常见问题
- 如遇显存不足，请调整 batch size 或使用 CPU。
- 如需 GPU 支持，请使用 `nvidia/cuda` 基础镜像并配置 `runtime: nvidia`。

---

## 参考
- [Docker 官方文档](https://docs.docker.com/)
- [docker-compose 官方文档](https://docs.docker.com/compose/)
