# Config registration workaround (unknown model_type on older transformers)

vLLM builds its `ModelConfig` by calling `transformers.AutoConfig.from_pretrained(model)`
**before** the vLLM-Neuron worker imports the model registry. If the container's
transformers is too old to know the architecture's `model_type`, this raises
`Unknown model type` and the model never loads.

## Fix
1. Define a minimal `PretrainedConfig` subclass that captures the fields vLLM and your
   `Config.from_configs` read. `model_type` must equal the `model_type` in the
   text-only `config.json` you serve (e.g. `"<arch>_text"`). PretrainedConfig stores
   any extra kwargs as attributes, so you only need to declare the ones you read plus
   the standard ones vLLM uses (vocab_size, hidden_size, num_hidden_layers,
   num_attention_heads, num_key_value_heads, head_dim, max_position_embeddings, the
   token ids, tie_word_embeddings).
2. `AutoConfig.register("<arch>_text", YourTextConfig)` — wrap in try/except ValueError
   (idempotent; native support may already exist).
3. **Register early.** Put the registration call in the plugin's `register()` in
   `vllm_neuron/__init__.py` (runs at platform activation, before `ModelConfig`):
   ```python
   from vllm_neuron.model.<arch>.hf_config import register_<arch>_configs
   register_<arch>_configs()
   ```
   Importing it only from the model package's `__init__` is too late (that import is
   triggered by the registry at worker init, after `ModelConfig`).

## Serve a text-only model dir (bypass multimodal)
For a multimodal checkpoint whose top-level `architectures` is
`*ForConditionalGeneration`, don't register that arch (it pulls vLLM's image-processor
path). Instead build a local dir:
- symlink/copy the original `*.safetensors`, `*.safetensors.index.json`, `tokenizer*`,
  `generation_config.json`;
- write a new `config.json` = the original `text_config` flattened to top level, with
  `architectures=["<Arch>ForCausalLM"]`, `model_type="<arch>_text"`,
  `tie_word_embeddings` carried over.
- Patch tokenizer_config.json quirks for the older transformers if needed (e.g.
  `extra_special_tokens` as a list → `{}`).
- Weight keys keep their original prefix (e.g. `model.language_model.layers.N.`), so
  `load_weights` mappings must use that prefix.
