import argparse
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

API_KEY_HF = os.environ["API_KEY_HF"]

login(API_KEY_HF)


def load_models(model_name, cache_dir):

    if model_name == 'llama-3':
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", token=API_KEY_HF)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-70B-Instruct",
            device_map='auto',
            cache_dir=cache_dir
        )

        messages = [
            {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a thug"},
            {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
        ]

        print(messages)

        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        generated_ids = model.generate(
            model_inputs,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_new_tokens=300
        )

        response = tokenizer.batch_decode(
            generated_ids[:, model_inputs.shape[1]:],
            skip_special_tokens=True
        )[0]

        print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/data/sedovaa20dm/models_cached/")
    args = parser.parse_args()
    load_models(model_name="llama-3", cache_dir=args.cache_dir)
