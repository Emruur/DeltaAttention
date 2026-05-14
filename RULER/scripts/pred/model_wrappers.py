# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import torch
from typing import Dict, List, Optional


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results


def _configure_globvr(globVR, experiment: str):
    import os
    if experiment == "baseline":
        globVR.delta_pf_key_on = 0
        globVR.delta_mlp = "Regular"
        globVR.flash = True
    elif experiment.startswith("row_delta"):
        globVR.delta_pf_key_on = 1
        globVR.delta_type = "row"
        globVR.scale = float(os.environ.get("DELTA_SCALE", "0.05"))
        globVR.delta_mlp = "Regular"
        globVR.row_delta_threshold = float(os.environ.get("DELTA_ROW_THRESH", "15"))
        globVR.row_similarity_metric = os.environ.get("DELTA_ROW_SIM", "cos")
        globVR.chunk_size = int(os.environ.get("DELTA_CHUNK_SIZE", "512"))
        globVR.divide_to = 0
        globVR.flash = True
        globVR.delta_decode = False
        globVR.dense_window_size = int(os.environ.get("DELTA_DENSE_WINDOW", "0"))


class DeltaLlamaModel:
    """Loads the custom DeltaLlama with globVR-controlled attention for RULER evaluation."""

    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        import os
        import sys

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import globVR
        from modeling_llama import LlamaForCausalLM, LlamaConfig
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        AutoConfig.register("llama", LlamaConfig, exist_ok=True)
        AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

        experiment = os.environ.get("DELTA_EXPERIMENT", "baseline")
        _configure_globvr(globVR, experiment)
        print(f"[DeltaLlama] experiment={experiment}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.model.eval()

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop("stop")

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=131072
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.generation_kwargs)

        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        results = []
        for text, prompt in zip(generated_texts, prompts):
            tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")
            decoded_prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(decoded_prompt):
                text = text[len(decoded_prompt):]
            if self.stop:
                for s in self.stop:
                    text = text.split(s)[0]
            results.append({"text": [text]})

        return results


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
