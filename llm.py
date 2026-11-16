import os
from collections.abc import Callable
from dotenv import load_dotenv
from groq import Groq
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam

load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
groq_chat_model = os.environ["GROQ_CHAT_MODEL"]


def _build_messages(user_prompt: str, system_prompt: str):
    return [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt),
    ]


def _stream_response(
    user_prompt: str,
    system_prompt: str,
    on_token: Callable[[str], None],
) -> str:
    """Stream a completion, calling on_token(delta) for every text fragment.

    Returns the full concatenated text so callers can use the result exactly
    like the non-streaming path (memory storage, citation parsing, ...).
    """
    stream = groq_client.chat.completions.create(
        model=groq_chat_model,
        messages=_build_messages(user_prompt, system_prompt),
        stream=True,
    )
    parts: list[str] = []
    usage = None
    for chunk in stream:
        x_groq = getattr(chunk, "x_groq", None)
        if x_groq is not None and getattr(x_groq, "usage", None):
            usage = x_groq.usage

        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            on_token(delta)

    if usage:
        print(f"[tokens] input={usage.prompt_tokens}  output={usage.completion_tokens}  total={usage.total_tokens}")
    return "".join(parts)


def get_response_from_llm(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Single LLM entry point.

    When `on_token` is given the response is streamed and each text fragment is
    passed to the callback; the full text is still returned at the end.
    """
    if on_token is not None:
        return _stream_response(user_prompt, system_prompt, on_token)

    response = groq_client.chat.completions.create(
        model=groq_chat_model,
        messages=_build_messages(user_prompt, system_prompt),
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