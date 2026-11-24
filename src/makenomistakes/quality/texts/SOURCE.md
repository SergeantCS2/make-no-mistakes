# The quality tune's texts

`llama-perplexity` reads one of these for a quality tune: the use's text, pinned by its sha256
in `quality/texts.py` (a changed byte is another text, and its scores never compare with the
old ones).

## prose.txt (chat and vision instances)

- What: the first 97,897 bytes (96 KiB, cut at the last line end) of WikiText-2's raw test
  split, `wikitext-2-raw/wiki.test.raw`.
- From: the zip llama.cpp's own `get-wikitext-2.sh` script fetches,
  `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`, read
  2026-11-14 (the whole test file: 1,290,590 bytes, sha256
  `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`).
- Who: WikiText, Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher, "Pointer
  Sentinel Mixture Models" (2016), from Wikipedia's Good and Featured articles.
- Licence: Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0,
  https://creativecommons.org/licenses/by-sa/3.0/), as WikiText is published; this file is
  an unmodified excerpt and stays under that licence, apart from mnm's MIT code.

## code.txt (coding instances)

- What: 98,257 bytes of makenomistakes' own Python source at version 5.0.0, each file
  under a `# ---- <path> ----` line: `fit/estimate.py`, `router_queue.py`, `tune/search.py`,
  `plan_types.py` and the first 459 lines of `fit/profiles.py`.
- Why mnm's own: a code text with a licence mnm can ship, and newer than any model's training
  data. The source moves on; this file does not.
- Licence: MIT (`LICENSE`), like the rest of mnm.
