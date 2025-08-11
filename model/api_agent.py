from openai import OpenAI
from typing import List
from model.base_agent import LLMAgent
import base64
import traceback
import time

def encode_image(image_path):
    if image_path:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        return "No image inputs"

class APIAgent(LLMAgent):
    def __init__(self, model_name, temperature=0) -> None:
        super().__init__(model_name, temperature)

        if "o3" in model_name or "gpt-5" in model_name or "o1" in model_name or "qwq" in model_name.lower() or "qvq" in model_name.lower() or "reasoner" in model_name.lower():
            self.max_tokens = 8192
        else:
            self.max_tokens = 2048

        if model_name in [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-3.5-turbo",
            "o3-mini-2025-01-31",
            "o1-2024-12-17",
            "o1-preview-2024-09-12",
            "o1-mini-2024-09-12",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
            "gpt-4o-2024-05-13",

            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",

            "gemini-1.5-pro",
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash",
            "gemini-2.0-pro-exp-02-05",
        ]:
            print("OpenAI")
            self.client = OpenAI(
                api_key="open_ai_api_key",
                base_url="https://api.openai.com/v1",
            )
        elif model_name in [
            "deepseek-chat",
            "deepseek-reasoner",
        ]:
            print("DeepSeek")
            self.client = OpenAI(
                api_key="[API_KEY]",
                base_url="https://api.deepseek.com/v1",
            )
        else:
            raise ValueError("Model not supported")

    def get_response(self, messages: List[dict]) -> str:
        if ("gpt-5" in self.model_name) or ("o3" in self.model_name) or ("o1" in self.model_name) or ("deepseek-reasoner" in self.model_name):
            messages = [m for m in messages if m["role"] != "system"]
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        max_completion_tokens=self.max_tokens,
                        seed=0
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    time.sleep(1)
                    response = "No answer provided."
        else:
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        temperature=self.temperature,
                        
                        max_tokens=self.max_tokens,
                        logprobs=True,
                        seed=0,
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    if "bad_response_status_code" in str(e):
                        print("Bad Response")
                        response = "No answer provided: bad_response."
                        break
                    elif "content_filter" in str(e):
                        print("Content Filter")
                        response = "No answer provided: content_filter."
                        break
                    else:
                        print(e)
                        print(traceback.format_exc())
                        time.sleep(1)
                        response = "No answer provided."
        try:
            log_probs = completion.choices[0].logprobs.content
            log_probs = [token_logprob.logprob for token_logprob in log_probs]
        except Exception as e:
            log_probs = []
        return response, log_probs

    def image_content(self, img_path: str) -> dict:
        img_path = img_path.strip()
        if img_path.startswith("http"):
            return {"type": "image_url", "image_url": {"url": img_path}}
        else:
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(img_path)}"}}
