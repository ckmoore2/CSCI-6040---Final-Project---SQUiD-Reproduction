from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')
import json
import logging
import re
import random
import numpy as np
from anthropic import Anthropic
random.seed(42)
np.random.seed(42)
import transformers
import torch
import time
import os
import json
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import mysql.connector
from mysql.connector import Error
import re
from tabulate import tabulate
from colorama import Fore, Style
 
class Model:
 
    def __init__(self,llm):
            self.llm = llm
            if llm=="openai":
                self.client = OpenAI(api_key="")
            elif llm=="llama":
                self.model, self.tokenizer, self.device = self.llama_model_init()
            elif llm=="mistral":
                self.model, self.tokenizer, self.device = self.mistral_model_init()
            elif llm=="deepseek":
                self.client = OpenAI(api_key="", base_url="https://api.deepseek.com")
            elif "qwen" in llm.lower():
                llm = "Qwen3-8B"
                self.llm = llm
                supported_models = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
                if llm in supported_models:
                    self.model, self.tokenizer, self.device = self.qwen_model_init(llm)
                else:
                    raise ValueError(f"Model {llm} is not supported. Supported models are: {supported_models}")
            elif llm=="claude":
                self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
 
    def qwen_model_init(self, model_name):
        device = torch.device("cuda:0")
        model_name = "Qwen/{}".format(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.eos_token = tokenizer.pad_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto").to(device)
        print("{} model loaded".format(model_name))
        return model, tokenizer, device
 
    def llama_model_init(self):
        device = torch.device("cuda:0")
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto').to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.eos_token = tokenizer.pad_token
        print("model loaded")
        return model, tokenizer, device
 
    def mistral_model_init(self):
        device = torch.device("cuda:0")
        model_name = "mistralai/Mistral-7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        return model, tokenizer, device
 
    def predict(self, prompt, user_prompt="", qwen_return_thinking=False):
        if self.llm == "llama":
            model, tokenizer = self.model, self.tokenizer
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
            generate_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False, temperature=1.0, top_k=1, top_p=1.0, repetition_penalty=1.0, num_beams=1)
            generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
            return tokenizer.decode(generate_ids)
 
        elif "qwen" in self.llm.lower():
            model, tokenizer = self.model, self.tokenizer
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            if qwen_return_thinking:
                content = thinking_content + content
            return content
 
        elif self.llm == "mistral":
            model, tokenizer = self.model, self.tokenizer
            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
            generate_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
            generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
            return tokenizer.decode(generate_ids)
 
        elif self.llm == "openai":
            completion = self.client.chat.completions.create(
                model="gpt-4o", store=True,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_prompt}],
                temperature=0, top_p=0, stop=None
            )
            return str(completion.choices[0].message.content)
 
        elif self.llm == "deepseek":
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_prompt}],
                stream=False
            )
            return str(response.choices[0].message.content)
 
        elif self.llm == "claude":
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                temperature=0,
                system=prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return str(response.content[0].text)
