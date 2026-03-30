# SmolVLM 推理服务 API 说明

基于 FastAPI 的图像 + 文本多模态推理接口（与 `trainer/infer.py` 同一套加载与生成逻辑）。响应为**一次性 JSON**，不支持流式输出。

## 基础信息

| 项目 | 说明 |
|------|------|
| 默认地址 | `http://127.0.0.1:8000`（可通过启动参数修改 host/port） |
| 协议 | HTTP/HTTPS |
| 认证 | 无（若需对外暴露请自行加反向代理与鉴权） |
| 交互式文档 | 服务启动后访问 [`/docs`](http://127.0.0.1:8000/docs)（Swagger UI）、[`/redoc`](http://127.0.0.1:8000/redoc) |

## 启动前准备

进程启动时会**加载一次**底座模型与 LoRA，必须通过环境变量或命令行传入（见下）。

| 环境变量 | 是否必填 | 说明 |
|----------|----------|------|
| `SMOL_MODEL_PATH` | 是 | 底座模型：Hugging Face 模型 ID（如 `HuggingFaceTB/SmolVLM-256M-Instruct`）或本地权重目录的绝对/相对路径 |
| `SMOL_ADAPTER_PATH` | 是 | PEFT LoRA 适配器目录（与 `trainer/train.py` 的 `--output_dir` 一致） |
| `SMOL_DEVICE` | 否 | 强制设备：`cuda` / `mps` / `cpu`；不设置则自动选择 |

## 启动方式

在仓库根目录执行（示例）：

```bash
python -m server.main \
  --model-path HuggingFaceTB/SmolVLM-256M-Instruct \
  --adapter-path /path/to/lora-output
```

可选参数：`--host`（默认 `0.0.0.0`）、`--port`（默认 `8000`）、`--device`（`cuda` / `mps` / `cpu`）。

也可先导出环境变量再启动 uvicorn：

```bash
export SMOL_MODEL_PATH=HuggingFaceTB/SmolVLM-256M-Instruct
export SMOL_ADAPTER_PATH=/path/to/lora-output
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

---

## 接口列表

### `GET /health`

健康检查，不加载模型时也可用于探活（若应用已启动）。

**响应**

| HTTP | 正文示例 |
|------|----------|
| 200 | `{"status":"ok"}` |

---

### `POST /infer`

上传一张图片与一段用户提示词，返回模型生成的文本（非流式）。

**Content-Type**

`multipart/form-data`

**表单字段**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | 是 | 用户文本提示（与 CLI `infer.py` 中传入模型的文本一致） |
| `image` | file | 是 | 图片文件（常见格式如 JPEG、PNG、WebP 等，服务端会转为 RGB） |
| `max_new_tokens` | integer | 否 | 最大生成 token 数，默认 `512`，合法范围 **1～8192** |

**成功响应**

| HTTP | 说明 |
|------|------|
| 200 | JSON 对象 |

**响应体 JSON 字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 从完整解码结果中提取的「助手」段落（与 `trainer/infer.py` 中 `extract_assistant` 一致） |
| `raw` | string | 整段解码文本（未截断到 Assistant 之后） |

示例：

```json
{
  "text": "move your hand slightly left, then grasp.",
  "raw": "User: ... Assistant: move your hand slightly left, then grasp."
}
```

**错误响应**

| HTTP | 典型原因 |
|------|----------|
| 400 | `prompt`/`image` 缺失、图片为空、无法解析为图片、`max_new_tokens` 超出范围 |
| 500 | 推理过程异常（`detail` 为错误信息字符串） |
| 503 | 模型未就绪（不应在正常运行中发生） |

---

## 请求示例

**cURL**

```bash
curl -s -X POST "http://127.0.0.1:8000/infer" \
  -F "prompt=What should the user do next?" \
  -F "image=@/path/to/image.jpg" \
  -F "max_new_tokens=256"
```

**Python（requests）**

```python
import requests

url = "http://127.0.0.1:8000/infer"
with open("image.jpg", "rb") as f:
    r = requests.post(
        url,
        files={"image": ("image.jpg", f, "image/jpeg")},
        data={"prompt": "Your prompt here", "max_new_tokens": "512"},
        timeout=300,
    )
r.raise_for_status()
print(r.json()["text"])
```

### 自带测试脚本

在**已启动**服务的前提下，可在仓库根目录执行（需已安装 `requests`）：

```bash
python server/test_api.py --image /path/to/image.jpg
```

仅探测健康检查：`python server/test_api.py --health-only`。更多参数见 `python server/test_api.py --help`。

---

## 注意事项

1. 模型与 LoRA 在进程内常驻内存，**多 worker** 时需重复占用显存；生产环境请按需设置 `uvicorn` worker 数量。
2. 大图片会先完整读入内存，超大请求请在前端或网关做限流与大小限制。
3. 与 `trainer/infer.py` 相同，底座路径可为 Hub ID；本地路径建议使用绝对路径或确保当前工作目录正确。
