import os
from dotenv import load_dotenv
from groq import Groq
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam

load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
groq_chat_model = os.environ["GROQ_CHAT_MODEL"]


def get_response_from_llm(user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    messages: list[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt),
    ]
    response = groq_client.chat.completions.create(
        model=groq_chat_model,
        messages=messages,
    )
    usage = response.usage
    print(f"[tokens] input={usage.prompt_tokens}  output={usage.completion_tokens}  total={usage.total_tokens}")
    return response.choices[0].message.content


def strip_fences(text: str) -> str:
    """Strip markdown code fences that LLMs sometimes add despite instructions."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()
