# Boss Agent

Boss Agent 是一个本地运行的求职流程助手。它通过对话理解求职需求，调用本机浏览器搜索 Boss 直聘职位，读取职位详情并进行匹配；职位搜索可以继续执行自动投递，职位推荐只返回推荐结果。

![系统架构](docs/architecture.png)

## 能做什么

- 从自然语言中提取职位、城市、薪资、经验、学历和排除条件。
- 自动区分职位搜索、职位推荐、简历分析、职位分析、闲聊和信息不明确的输入。简历分析、职位分析和闲聊不会启动浏览器；职位推荐会搜索并匹配职位，但不会自动投递。
- 通过本机浏览器检查登录、搜索职位、读取详情并执行自动投递。登录状态、解析进度、匹配进度和投递结果会实时反馈到前端。
- 用 Qdrant 保存职位向量；服务不可用时回退到本地余弦匹配。
- 在 SQLite 中保存配置、会话、任务进度和投递状态，支持任务恢复、会话删除和投递幂等控制。
- 支持从对话页面暂停/继续任务；浏览器或 SSE 连接断开时，会请求停止对应的后台任务。
- 可选监听 Boss 聊天页面，并根据已有对话生成自动回复；涉及薪资、面试时间、隐私和附件时只生成建议，不替用户做决定。

![任务流程](docs/workflow.png)

## 运行

要求 Python 3.10 及以上。建议使用虚拟环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r request.txt
python main.py
```

浏览器打开 [http://127.0.0.1:5001](http://127.0.0.1:5001)。

上传简历支持 `.pdf`、`.docx` 和 `.doc`，单个文件最大 30MB。解析 `.pdf` 和 `.docx` 使用 Python 依赖；解析旧版 `.doc` 还需要本机安装 LibreOffice。若不方便安装 LibreOffice，可以先将文件另存为 `.docx`。

首次使用时，在页面设置中分别填写对话模型和 Embedding 模型的 API Key、Base URL、模型名称。Embedding 模型需要返回向量，例如 `text-embedding-3-small`。缺少必要模型配置时，前端会显示任务错误，不会静默执行。

首次搜索时，程序会打开或连接已有的 Boss 直聘浏览器实例。若尚未登录或出现验证码，需要在该浏览器窗口中手动完成；前端会显示等待登录状态，登录完成后任务自动继续。

## 任务行为

- **职位搜索**：解析筛选条件，读取职位详情，排除关键词后进行向量匹配，并对匹配职位自动投递。
- **职位推荐**：可以根据用户条件或已上传简历提取职位方向，搜索并返回匹配结果，不执行投递。
- **简历分析**：支持 PDF、DOCX、DOC 简历解析，输出经历概括、优势、风险和修改建议；未上传简历时会先提示上传。
- **职位分析**：根据用户提供的职位描述生成分析，不启动 Boss 浏览器。
- **闲聊或需求不明确**：直接回复或要求补充信息，不进入登录检查。

职位详情读取和匹配在默认 5 分钟后超时。若详情读取已经安全结束，系统会使用当前已经完成的匹配结果继续投递；若浏览器仍被详情读取占用，则停止本次投递，避免并发操作同一浏览器页面。

前端通过 SSE 接收任务状态。任务处于登录等待、暂停或处理中时会持续发送状态和心跳；网络连接断开后，后台任务会在下一个安全检查点取消。暂停不会强行中断当前浏览器操作，任务会在安全检查点暂停。

## 配置

页面设置会写入 `data/config.db`。也可以用环境变量提供初始值：

```bash
export chat_openai_api_key="your-chat-key"
export chat_openai_base_url="https://api.openai.com/v1"
export chat_openai_model="gpt-4o-mini"
export embedding_openai_api_key="your-embedding-key"
export embedding_openai_base_url="https://api.openai.com/v1"
export embedding_openai_model="text-embedding-3-small"
export auto_reply="false"
export qdrant_enabled="true"
```

页面设置支持以下配置：

- 对话模型：`chat_openai_api_key`、`chat_openai_base_url`、`chat_openai_model`。
- Embedding 模型：`embedding_openai_api_key`、`embedding_openai_base_url`、`embedding_openai_model`。
- Qdrant：`qdrant_enabled`、`qdrant_url`、`qdrant_api_key`、`qdrant_path`、`qdrant_collection`。
- 自动回复：`auto_reply`。

Qdrant 默认启用，未填写 `qdrant_url` 时使用本地持久化目录 `data/qdrant/`；填写远程地址后改为连接远程 Qdrant，受保护实例再填写 `qdrant_api_key`。设置 `qdrant_enabled=false` 可关闭 Qdrant，系统会回退到本地向量匹配。

`auto_reply=true` 会启动独立的 Boss 聊天监听线程。浏览器尚未就绪时监听器会等待，关闭配置后会停止监听。

职位详情读取和匹配默认最多运行 5 分钟，可通过环境变量调整：

```bash
export JOB_PIPELINE_TIMEOUT_SECONDS=300
```

投递网络异常默认最多尝试 3 次；已经记录为 `sending` 或 `unknown` 的职位，
超过 30 分钟租约后允许再次尝试，避免进程中断后永久卡住。由于网络异常时远端可能已经收到请求，
重试和租约恢复仍不能完全排除远端重复投递的可能性。

## 筛选映射

自然语言筛选项先由对话模型映射到标准选项名称，后端再从 `data/` 下的 JSON 码表读取 Boss 参数：

- `data/money_codes.json`：薪资。
- `data/experience_codes.json`：经验。
- `data/degree_codes.json`：学历。
- `data/scale_codes.json`：公司规模。
- `data/city_codes.json`：城市。

后端对码表名称执行精确匹配，不会自行猜测或生成编码。排除关键词是职位内容过滤条件，会对职位名称、职位描述和行业字段进行不区分大小写的包含匹配；它不是对整条用户输入做正则匹配。

## 项目结构

- `main.py`：启动 Web 服务和可选聊天监听。
- `web_flask.py`：页面、配置、会话和任务接口。
- `body.py`：意图解析、简历解析和 LangGraph 流程。
- `browser.py`：Boss 页面操作。
- `job.py`：详情采集、匹配和投递管线。
- `matcher.py`：码表读取、排除关键词过滤和匹配入口。
- `vector_store.py`：Qdrant 索引与召回。
- `conversation_store.py`、`delivery_store.py`：会话、任务和投递状态存储。
- `sql.py`：SQLite 连接、事务、线程锁和关闭逻辑的公共封装。
- `data/*_codes.json`：职位筛选与城市映射码表。

## 接口

页面由 Flask 应用提供，主要接口包括：

- `GET /api/health`：服务健康状态。
- `GET /api/config`、`POST /api/config`：读取和保存页面配置。
- `GET /api/conversations`、`POST /api/conversations`：查询和创建会话。
- `GET /api/conversations/<conversation_id>/messages`：读取会话消息。
- `DELETE /api/conversations/<conversation_id>`：删除会话及关联消息、任务快照和上传简历。
- `POST /api/chat`：提交聊天任务并通过 SSE 返回状态和结果。
- `GET /api/tasks`、`GET /api/tasks/<task_id>`：查询可恢复任务或任务快照。
- `POST /api/tasks/<task_id>/pause`、`POST /api/tasks/<task_id>/resume`：暂停或继续任务。

## 数据与限制

`.env`、SQLite 数据库、上传简历、浏览器用户目录、Qdrant 数据、运行日志和测试/调试数据都保留在本机，不提交到 Git。

默认本地数据位置：

- `data/config.db`：页面配置。
- `data/conversations.db`：会话、消息和任务快照。
- `data/deliveries.db`：职位投递状态和尝试次数。
- `data/resumes/`：上传的简历。
- `data/qdrant/`：本地 Qdrant 数据。
- `logs/`：运行日志。

程序依赖 Boss 页面结构和本机登录状态。页面改版、验证码、访问限制或账号风控都可能使采集中断；任务会保留已完成的进度，但不会绕过网站限制。
