from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
import anthropic
from groq import Groq
import os

load_dotenv()

def get_openai_gpt_completions(params):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    completion = client.chat.completions.create(
        n=params['n_completions'],
        model='gpt-4',
        messages=[
            {"role": "user", "content": params['system_prompt']},
            {"role": "user", "content": params['user_prompt']}
        ]
    )

    responses = {}
    for i, completion in enumerate(completion.choices):
        responses[i] = completion.message.content

    return responses

def get_google_gemini_completions(params):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model_name = params.get("model", "gemini-2.5-pro")
    model = genai.GenerativeModel(model_name)

    prompt = f"{params['system_prompt']}\n\n{params['user_prompt']}"

    result = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            candidate_count=params["n_completions"], 
            temperature=1.0,
        )
    )

    responses = {}

    for i, cand in enumerate(result.candidates):
        responses[i] = cand.content.parts[0].text

    return responses


from anthropic import APIError

def get_anthropic_claude_completions(params, max_retries=5):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    responses = {}

    for i in range(params["n_completions"]):
        delay = 1

        for attempt in range(max_retries):
            try:
                result = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": params["user_prompt"]}]
                )
                break  # success → leave retry loop

            except APIError as e:
                err_type = getattr(e, "error", {}).get("type")

                if err_type == "overloaded_error":
                    print(f"[{i}] Claude overloaded. Retry {attempt+1}/{max_retries} in {delay}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 16)  # exponential backoff, capped at 16s
                    continue
                else:
                    raise  # real error → surface immediately

        else:
            # If we exhausted all retries
            raise RuntimeError(f"Claude still overloaded after {max_retries} retries for completion {i}")

        print("GT")

        # Extract text safely
        full_text = ""
        for block in result.content:
            if block.type == "text":
                full_text += block.text

        responses[i] = full_text

    return responses


def get_groq_completions(params):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    responses = {}

    for i in range(params["n_completions"]):
        result = client.chat.completions.create(
            model=params.get("model", "moonshotai/kimi-k2-instruct-0905"),  # Default recommended model
            messages=[
                {"role": "user", "content": params['system_prompt']},
                {"role": "user", "content": params['user_prompt']}
            ],
            temperature=1.0,
            max_tokens=4096,
            stream=False
        )

        responses[i] = result.choices[0].message.content

    return responses