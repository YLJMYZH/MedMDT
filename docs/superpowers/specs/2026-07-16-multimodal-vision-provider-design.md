# MedMDT 图像理解 Provider 接入设计

## 1. 背景

MedMDT 已将会诊、知识处理、图片分析和 Embedding 分为独立配置。图片分析通过 `ImageParser` 把医学图片发送给 LangChain `BaseChatModel`，但当前实现存在以下问题：

- 图片消息固定使用 OpenAI `image_url` 格式，而各 provider 的文字模型工厂混用了 OpenAI 兼容接口和厂商原生集成，视觉兼容性没有明确边界。
- 图片 MIME 被硬编码为 `image/png`，JPEG、WebP 等输入会被错误标记。
- 图片分析配置可选择所有 provider，且前端默认值是当前不支持图片输入的 `deepseek-chat`。
- 现有连接测试只发送文字，不能验证具体模型是否真正接受图片。
- `response.content` 被假设为字符串，不能可靠处理 LangChain 内容块响应。

本设计只覆盖图像理解：一张或多张图片输入，结构化文本结果输出。不包括视频、音频、图片生成、视觉工具调用和临床准确率评估。

## 2. 目标

- 为图片分析建立独立于文字模型的 provider 路由。
- 保持 `ImageParser` 与厂商 SDK 解耦。
- 统一图片输入格式，同时允许视觉和文字场景为同一 provider 选择不同传输接口。
- 在设置页准确表达 provider 级视觉能力，并通过真实图片请求验证模型级能力。
- 保留 DeepSeek provider，当前明确标记视觉 API 不可用，未来通过注册表配置启用。
- 提供不泄露密钥和影像内容的分类错误与日志。

## 3. 非目标

- 不统一替换现有文字模型的原生 LangChain 集成。
- 不通过模型名称白名单断言模型一定支持视觉。
- 不引入 `langchain-glm`。
- 不为每个厂商编写完整原生 SDK 客户端。
- 不在普通 CI 中调用收费的外部模型 API。
- 不将“接口调用成功”等同于医学诊断有效。

## 4. 方案选择

### 4.1 采用方案：独立视觉工厂

文字模型继续根据厂商能力使用现有集成；视觉模型通过独立的 `create_vision_model()` 路由。对于官方支持 OpenAI Chat Completions 图像消息格式的厂商，视觉调用统一使用 `ChatOpenAI` 和厂商 `base_url`。Anthropic 使用 `ChatAnthropic`，由其 LangChain 集成完成消息转换。

该方案在当前只需要图像理解的范围内，兼顾统一消息格式、低维护成本和未来扩展能力。

### 4.2 未采用：全部改成 OpenAI 兼容接口

该方案代码最少，但会削弱文字、工具调用、思考参数等厂商原生能力，Anthropic 仍需单独处理，因此不修改现有文字模型策略。

### 4.3 未采用：所有厂商均使用原生 SDK

该方案控制力最强，但需要分别维护图片消息、响应和错误转换。当前需求只包含图像理解，维护成本与收益不匹配。

## 5. Provider 路由

### 5.1 单一注册表

Provider 元数据、文字工厂、视觉工厂、默认 Base URL 和视觉状态集中在同一个注册表中。设置 API、普通模型工厂和视觉模型工厂都从该注册表读取，避免重复维护。

概念结构：

```python
ProviderSpec(
    key="qwen",
    label="Qwen (通义千问)",
    chat_factory=create_tongyi_chat,
    vision_factory=create_openai_compatible_vision,
    vision_status="supported",
    vision_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

`vision_status` 使用以下状态：

- `supported`：provider 有已确认的图片输入协议。
- `unavailable`：provider 保留，但当前官方 API 不提供图片输入。
- `unknown`：自定义 OpenAI 兼容端点，必须实测。

### 5.2 路由矩阵

| Provider | 文字调用 | 图像理解调用 | 当前视觉状态 |
|---|---|---|---|
| OpenAI | `ChatOpenAI` | `ChatOpenAI` | supported |
| Anthropic | `ChatAnthropic` | `ChatAnthropic` | supported |
| Qwen | `ChatTongyi` | `ChatOpenAI` + DashScope OpenAI-compatible URL | supported |
| Zhipu | `ChatZhipuAI` | `ChatOpenAI` + BigModel OpenAI-compatible URL | supported |
| Moonshot | `ChatOpenAI` + Moonshot URL | 同左 | supported |
| DeepSeek | `ChatDeepSeek` | 无；抛出明确能力错误 | unavailable |
| Custom | `ChatOpenAI` + 自定义 URL | 同左 | unknown |

第三方 OpenAI 兼容视觉调用固定使用 Chat Completions 语义，不启用仅 OpenAI 官方支持的 Responses API 特性。

选择 OpenAI 兼容视觉接口的依据：

- [Qwen OpenAI-compatible Chat 文档](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)定义了 `image_url` 和 Base64 Data URL。
- [智谱 GLM-4.6V 文档](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-4.6v)使用 Chat Completions `image_url` 内容块。
- [Kimi 视觉模型文档](https://platform.kimi.com/docs/guide/use-kimi-vision-model)使用 OpenAI SDK、厂商 Base URL 和 `image_url` 内容块。
- [Anthropic 视觉文档](https://docs.anthropic.com/zh-CN/docs/build-with-claude/vision)使用原生 `image` 内容块；`ChatAnthropic` 负责从 LangChain/OpenAI 风格块转换。

## 6. 组件职责

### 6.1 `create_chat_model()`

- 保持现有公开接口和文字调用策略。
- 供会诊、知识提取、检索和文字连接测试使用。
- 不因为视觉需求而把 Qwen、智谱或 DeepSeek 的文字路径改回通用兼容接口。

### 6.2 `create_vision_model()`

- 接收 `provider`、`model`、`api_key`、可选 `base_url` 和通用生成参数。
- 根据统一注册表选择视觉传输实现。
- 对 `unavailable` provider 在发起网络请求前抛出 `VisionProviderNotSupported`。
- Custom 必须提供 `base_url`，其兼容性由真实视觉测试确认。
- 返回 `BaseChatModel`，使 `ImageParser` 继续使用统一的 `.invoke()`。

### 6.3 `ImageParser`

- 验证图片字节非空且可以被 Pillow 解码。
- 根据图片内容识别 MIME，不依赖扩展名。
- 对通用格式直接构造 Data URL；DICOM 渲染结果及不通用格式转换为无损 PNG。
- 构造统一的文字加图片消息。
- 调用注入的视觉模型。
- 从字符串或 LangChain 内容块中提取最终文本。
- 清理 Markdown fence，并通过 `ImageAnalysisResult` 做 Pydantic 校验。
- 首次仅因 JSON 格式错误而失败时，最多执行一次纠正重试。

`ImageParser` 不读取 provider 名称，也不包含厂商分支。

### 6.4 设置 API 与前端

- 普通模型区域继续展示所有 provider。
- 图片分析区域读取注册表返回的视觉状态。
- DeepSeek 继续显示并标记“官方 API 暂不支持图像理解”，当前不可保存为有效视觉配置。
- Custom 标记“兼容性未知，需测试”。
- 不使用静态模型名白名单；provider 支持视觉不代表所选具体模型支持视觉。
- 普通文字连接测试和视觉能力测试使用独立端点及按钮。

## 7. 数据流与消息格式

```text
上传图片或 DICOM 渲染结果
        ↓
图片解码、格式识别和大小校验
        ↓
正确 MIME + Base64 Data URL
        ↓
HumanMessage(text + image_url)
        ↓
create_vision_model 选定的 BaseChatModel
        ↓
字符串或内容块响应
        ↓
文本归一化、JSON 清理、Pydantic 校验
        ↓
ImageAnalysisResult
```

统一输入消息：

```python
HumanMessage(
    content=[
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{encoded}",
            },
        },
    ]
)
```

结构化输出继续采用明确的 JSON 提示词和 Pydantic 校验，不依赖 `with_structured_output()` 或厂商特有的 `response_format`，以避免第三方兼容端点实现差异。

## 8. 图片处理规则

- 支持并正确标记 PNG、JPEG、WebP 和 GIF。
- DICOM 由现有 DICOM 解析器渲染为 PNG 后进入视觉链路。
- Pillow 可解码但不属于通用直传集合的静态图片转换为无损 PNG。
- 空文件、损坏文件、无法识别的文件在网络调用前拒绝。
- 图片超过配置的传输大小限制时明确报错，不静默降采样医学影像。
- Base64 数据、原始图片和患者上下文不得写入日志。

## 9. 视觉能力测试

新增独立视觉测试端点。它发送一张内置的小型有效测试图片，并验证：

1. API Key 和 Base URL 可用。
2. 所选模型接受图片内容块。
3. 模型返回文本内容。
4. 返回内容可通过图片分析结构校验。

保存配置不自动调用外部 API，避免意外费用。未测试配置在界面显示提示。视觉测试结果用于确认当前 provider、端点和具体模型组合，不持久化为永久能力事实。

## 10. 错误处理

错误按以下类别向上层暴露：

- `VisionProviderNotSupported`：provider 当前没有官方图像输入接口，例如 DeepSeek。
- `VisionModelNotSupported`：provider 有视觉接口，但所选模型拒绝图片。
- `InvalidImageError`：图片为空、损坏、格式不支持或超过限制。
- `VisionRequestError`：认证、限流、网络或厂商服务错误。
- `InvalidVisionResponse`：响应中没有可用文本，或最终不能通过结构校验。

只有首次 JSON 格式错误可以自动纠正重试一次。认证失败、限流、模型不支持和图片非法不做格式纠正重试。日志记录 provider、model、错误类别和请求追踪 ID，不记录密钥、图片、Data URL 或临床背景。

## 11. 测试策略

### 11.1 视觉工厂单元测试

- 验证每个 provider 选择正确的 LangChain 类、Base URL 和参数。
- 验证第三方兼容端点不启用 Responses API 专有路径。
- 验证 DeepSeek 在网络调用前返回明确能力错误。
- 验证 Custom 缺少 Base URL 时失败。

### 11.2 图片消息测试

- 覆盖 PNG、JPEG、WebP 的 MIME 识别和 Data URL。
- 覆盖损坏、空文件、不通用格式转换和大小限制。
- 验证 DICOM 渲染后的 PNG 链路。

### 11.3 响应解析测试

- 覆盖字符串响应、LangChain 内容块和 Markdown JSON fence。
- 覆盖首次格式错误后的单次纠正重试。
- 覆盖最终响应无法解析的分类错误。

### 11.4 设置 API 测试

- 验证文字连接测试与视觉能力测试互不混用。
- 验证 provider 视觉状态正确返回。
- 验证 DeepSeek 显示但不能成为当前有效视觉配置。
- 验证 Custom 可配置并通过真实视觉请求判断兼容性。

### 11.5 可选外部契约测试

各厂商真实 API 测试通过环境变量提供密钥，并使用显式测试标记；默认 CI 跳过，避免费用和外部服务波动影响常规测试。

## 12. 完成标准

- 文字模型现有路由行为保持不变。
- 图片分析使用独立视觉工厂。
- OpenAI、Anthropic、Qwen、智谱和 Moonshot 能通过统一 `ImageParser` 发送图像理解请求。
- DeepSeek 保留在 provider 注册表，并对当前视觉调用给出明确提示。
- Custom 可配置，并能通过真实图片测试确认兼容性。
- 图片 MIME 不再硬编码。
- 字符串和内容块响应均能正确解析。
- 设置页能区分 provider 视觉状态和具体模型视觉测试结果。
- 自动化测试覆盖路由、图片、响应和设置 API；常规测试不依赖真实厂商 API。
- 技术连通性结果不被表述为临床诊断准确性证明。
