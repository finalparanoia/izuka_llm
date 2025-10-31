from typing import List, Dict
from loguru import logger
from fastapi import APIRouter, HTTPException

from izuka_llm.schemas.dto import ModelList, ModelInfo, ChatCompletionResponse, ChatCompletionRequest, ChatMessage, \
    ChatChoice, UsageInfo

router = APIRouter(prefix="/v1")


# --- 3. 核心 LLM 逻辑 ---
# 这里我们用一个简单的函数来模拟大语言模型的调用。
# 在真实场景中，这里会调用你的模型推理代码。

def fake_llm_call(messages: List[Dict[str, str]]) -> str:
    """
    一个模拟的 LLM 调用函数。
    它会获取最后一条用户消息，并生成一个模拟的回复。
    """
    # 找到最后一条用户消息
    last_user_message = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_message = msg["content"]
            break

    # 生成模拟回复
    if not last_user_message:
        return "你好！有什么可以帮助你的吗？"
    return f"这是一个模拟回复，针对你的问题：'{last_user_message}'。"


# --- 4. API 端点实现 ---

@router.get("/models", response_model=ModelList)
async def list_models():
    """
    列出所有可用的模型。
    这模仿了 OpenAI 的 `/v1/models` 端点。
    """
    # 在这里可以定义你希望暴露给客户端的模型列表
    available_models = [
        ModelInfo(id="gpt-3.5-turbo"),
        ModelInfo(id="gpt-4"),
        ModelInfo(id="custom-local-model"),
    ]
    return ModelList(data=available_models)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    创建一个聊天补全。
    这模仿了 OpenAI 的 `/v1/chat/completions` 端点。
    """
    # 检查请求的模型是否在我们的支持列表中
    supported_models = ["gpt-3.5-turbo", "gpt-4", "custom-local-model"]
    if request.model not in supported_models:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not found.")

    # 将 Pydantic 模型转换为字典，方便传递给 LLM 函数
    messages_dict = [msg.model_dump() for msg in request.messages]

    # 调用我们的模拟 LLM
    assistant_response_content = fake_llm_call(messages_dict)

    # 构建符合 OpenAI 格式的响应
    assistant_message = ChatMessage(role="assistant", content=assistant_response_content)
    choice = ChatChoice(index=0, message=assistant_message)

    # 模拟 token 使用情况
    prompt_tokens = sum(len(msg['content']) for msg in messages_dict)
    completion_tokens = len(assistant_response_content)
    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )

    response = ChatCompletionResponse(
        model=request.model,
        choices=[choice],
        usage=usage
    )
    logger.debug(response)
    return response


# 您原始的 `/model` 端点可以被保留或删除。
# 这里我们将其重定向到新的 `/v1/models` 端点，以保持兼容性。
@router.get("/model", include_in_schema=False)
async def redirect_model_api():
    """一个简单的重定向，将旧的 /model 请求导向新的 /v1/models"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/v1/models")

