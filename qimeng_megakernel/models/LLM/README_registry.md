# LLM model registry

- **registry.json** — Committed. Maps HuggingFace model ids to the model/hardware-specific preprocess backend. All LLMs use the shared `qimeng_megakernel.models.LLM.decode`; per-model decode entries are forbidden. Do not put local paths here.
- **registry.local.json** — Optional, **gitignored** (not uploaded). You create it to use a **local model path**.

## Using a local model

1. Copy `registry.local.json.example` to `registry.local.json` in this directory.
2. Add one line: your local path → model_key (the key must exist in `registry.json`, e.g. `llama-3.2-1b`):

   ```json
   {
     "/your/path/to/llama3.2_1b": "llama-3.2-1b"
   }
   ```

3. Run with that path: `--model /your/path/to/llama3.2_1b` or `LLM_MODEL=/your/path/to/llama3.2_1b`.

If you run with a local path and have not added it to `registry.local.json`, the error message will tell you to do this.
