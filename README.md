# Benckmark LLMs

Speed benchmark tool for LLMs, supporting LLaMA and HuggingFace model inferfaces.

## To Use

```
# Single GPU
python b2.py config.yaml

# For Original LLaMA
torchrun --nproc-per-node 2 b2.py config.yaml

# For Deepspeed
deepseep --num_gpus=1 b2.py config.yaml
# OR
CUDA_VISIBLE_DEVICES=0,1 deepseep b2.py config.yaml
```

## References

## APIs

### Hugging Face Generators

```python
model: LlamaForCausalLM
outputs: = model(input_ids)

model: LlamaForCausalLM
generate_ids = model.generate(input_ids, max_length, streamer)
```

Call stack:

```
LlamaForCausalLM.generate
-> GenerationMixin.generate
  -> GenerationMixin.greedy_search
    -> LlamaForCausalLM.forward
      -> LlamaModel.forward
      <- BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
    <- CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
  <- input_ids
<- input_ids
```

### LLaMA

```python
generator: LLaMA
generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )
```

Call stack:

```
LLaMA.generate()
  for cur_pos in range(start_pos, total_len):
      logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
      next_token = _sample(logits)
<- decoded_tokens
```

##

Benchmark:
输入: model
输出：times
选项：benckmark_first
benckmark_next
bss
check results

存csv, no eos

times = benckmark(wrap(model))

HF style

input_ids
max_length
model.generate()
times from streamer

llama style

time from wrapper




benckconfig

global_specs :



models :
  hf_baseline:
    name_or_path:
    launcher:
  deepspeed:
    name_or_path:
    launcher:
