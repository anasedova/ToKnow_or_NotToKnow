import os
import random
import openai

from tqdm import tqdm
from huggingface_hub import InferenceClient
from huggingface_hub import login

API_KEY_HF = os.environ["API_KEY_HF"]
API_KEY_GPT = os.environ["API_KEY_GPT"]

login(API_KEY_HF)

SMALL_MODELS_URL = "http://localhost:19082/v1/"
LLAMA_3_URL = "http://localhost:19082/v1/"
MIXTRAL_URL = "http://localhost:19082/v1/"
LLAMA_2_URL = "http://localhost:8080/v1/"

random.seed(42)


def send_prompt(prompts, hyperparams, processed_prompts=None):
    results = []
    for prompt in tqdm(tqdm(prompts)):

        # w/ and w/o system prompt
        # in the experiments for the EMNLP 2024 paper, no system prompts were used
        for system_prompt in [
            "",
            # "You are a helpful assistant. Ask clarification questions if instruction is unclear.",
            # "You are a helpful assistant. Pay attention that all the provided entities are ambiguous: "
            # "all of them are the companies named after something else"
        ]:

            try:
                # do not send this prompt again if we have already processed it
                if processed_prompts and {"system_prompt": system_prompt, "prompt": prompt} in processed_prompts:
                    continue

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response_text = model_request_via_inference_client(messages, hyperparams)
                response = {"system_prompt": system_prompt, "prompt": prompt, "response": response_text}
                results.append(response)

            except Exception as ex:
                print(str(ex))

    return results


def model_request(messages, hyperparams, model, tokenizer):

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    generated_ids = model.generate(model_inputs, pad_token_id=tokenizer.eos_token_id, do_sample=True, max_new_tokens=500)
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.shape[1]:], skip_special_tokens=True)[0]
    print(response)

    return response


# if model was run via docker on the server
def model_request_via_inference_client(messages, hyperparams):

    if hyperparams["model"] in ["llama-2-70b-chat-hf"]:
        prompt = f"""[INST]<<SYS>><</SYS>>{messages[0]['content']}[/INST]"""
        for message in messages[1:]:
            if message["role"] == "assistant":
                prompt += f"""{message['content']}"""
            else:
                prompt += f"""[INST]{message['content']}[/INST]"""

        client = InferenceClient(LLAMA_2_URL)
        return client.text_generation(
            prompt=prompt,
            max_new_tokens=600,
            details=True,
            temperature=hyperparams["temp"],
            do_sample=True
        )

    elif hyperparams["model"] in ["mixtral-8x7B-Instruct-v0.1"]:
        # remove system prompt which is not supported in mistral
        messages = list(filter(lambda x: x["role"] == "user", messages))

        client = openai.OpenAI(base_url=MIXTRAL_URL, api_key="EMPTY")
        response = client.chat.completions.create(
            model="tgi",
            messages=messages,
            temperature=hyperparams['temp'],
            seed=hyperparams['seed'],
            max_tokens=4000
        )

    # run LLama-3
    elif hyperparams["model"] == "llama-3-70b":

        client = openai.OpenAI(base_url=LLAMA_3_URL, api_key="EMPTY")
        response = client.chat.completions.create(
            model="tgi",
            messages=messages,
            temperature=hyperparams['temp'],
            seed=hyperparams['seed'],
            logprobs=True,
            max_tokens=4000,
            stop=[
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
                "<|reserved_special_token|>",
            ]
        )

    # run Gemma_instruct
    elif hyperparams["model"] == "gemma-1.1-7b-it":

        # remove system prompt which is not supported in gemma
        messages = list(filter(lambda x: x["role"] == "user", messages))

        client = openai.OpenAI(base_url=SMALL_MODELS_URL, api_key="EMPTY")
        response = client.chat.completions.create(
            model="tgi",
            messages=messages,
            temperature=hyperparams['temp'],
            seed=hyperparams['seed'],
            max_tokens=5000,
        )

    # run GPT
    elif hyperparams["model"] in ["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4", "gpt-4-turbo", "gpt-4o"]:

        client = openai.Client(api_key=API_KEY_GPT)
        response = client.chat.completions.create(
            model=hyperparams['model'],
            messages=messages,
            temperature=hyperparams['temp'],
            seed=hyperparams['seed']
        )

    elif hyperparams["model"] in ["mistral-7B-Instruct-v0.2"]:
        # remove system prompt which is not supported in gemma
        messages = list(filter(lambda x: x["role"] == "user", messages))

        client = openai.OpenAI(base_url=SMALL_MODELS_URL, api_key="EMPTY")
        response = client.chat.completions.create(
            model="tgi",
            messages=messages,
            temperature=hyperparams['temp'],
            seed=hyperparams['seed'],
            max_tokens=4000
        )

    else:
        raise ValueError(f"Unknown model! {hyperparams['model']}")

    return response.choices[0].message.content
