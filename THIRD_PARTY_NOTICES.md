# Third-party notices

make-no-mistakes is MIT-licensed (`LICENSE`). The engines and apps it runs (llama.cpp,
stable-diffusion.cpp, whisper.cpp, audio.cpp, vLLM, ComfyUI and its frontend, the workspace
apps) are separate programs in their own containers, under their own licences: mnm downloads
them from their publishers and copies none of their code.

What this repository holds or adapts from others:

| What | Where | From | Licence |
|---|---|---|---|
| An excerpt of WikiText-2's raw test split (the first 96 KiB), the text the quality tune measures chat and vision quants on | `src/makenomistakes/quality/texts/prose.txt` (provenance in `SOURCE.md` beside it) | WikiText (Merity, Xiong, Bradbury, Socher, "Pointer Sentinel Mixture Models", 2016), from Wikipedia's articles, via [ggml-org/ci](https://huggingface.co/datasets/ggml-org/ci) | CC BY-SA 3.0 ([the licence](https://creativecommons.org/licenses/by-sa/3.0/)); the excerpt stays under it |
| ComfyUI's partner node names (the API nodes mnm refuses) | `src/makenomistakes/comfy/partner_nodes.json` | recorded from ComfyUI v0.42.1's `/object_info` (data a running ComfyUI serves, not its code) | - |
| Hugging Face's hardware table (each part's name, memory, TFLOPS and how many users registered it), the list of cards the device table covers | `src/makenomistakes/fit/data/hf-hardware-2026-10-27.json` | exported from [huggingface.co/hardware](https://huggingface.co/hardware?view=table) on 2026-10-27 | - |
| ACE-Step 1.5's dependency pins, the constraints the style trainer's locks are compiled against | `src/makenomistakes/studio/style_docker/acestep-uvlock.txt` | exported from the `uv.lock` of [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) at ca1e85f | MIT |
| The approach of the UI-to-API workflow converter (links resolved through reroutes, bypassed nodes and subgraph boundaries); mnm's code is its own | `src/makenomistakes/comfy/convert.py` | [SethRobinson/comfyui-workflow-to-api-converter-endpoint](https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint) | Unlicense (public domain) |
| The VT323 and IBM Plex Mono fonts | `src/makenomistakes/web/static/fonts/` (each with its licence) | Peter Hull; IBM | SIL Open Font License 1.1 |
| htmx 1.9.12 | `src/makenomistakes/web/static/htmx-1.9.12.min.js` (its licence beside it) | [bigskysoftware/htmx](https://github.com/bigskysoftware/htmx) | Zero-Clause BSD |
