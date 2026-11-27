---
license: mit
base_model: imageomics/bioclip-2
library_name: coreml
pipeline_tag: zero-shot-image-classification
tags:
  - coreml
  - apple-silicon
  - bioclip
  - biology
  - tree-of-life
  - clip
---

# BioCLIP 2 image encoder — Core ML (Apple Silicon)

A Core ML conversion of the **image tower** of
[imageomics/bioclip-2](https://huggingface.co/imageomics/bioclip-2)
(ViT-L/14, revision `2957b322090f9cb17ae72c71981c7218a28d81e0`), for
fast on-device encoding on Apple Silicon Macs (GPU + Neural Engine).

It is used by [BotaniSense for Immich](https://github.com/cxcxcxcx/botanisense-for-immich)
when run natively on macOS (`ENGINE=auto` / `ENGINE=coreml`), but it is a
plain Core ML model and works anywhere you need BioCLIP 2 image embeddings.

This repository contains **only the image encoder**. The text side is
unchanged: classify by comparing the embeddings against the original
BioCLIP 2 text embeddings (e.g. `txt_emb_species.npy` from
[imageomics/TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M),
as [pybioclip](https://github.com/Imageomics/pybioclip) does).

## Files

| File | What |
|---|---|
| `encoder_b16.mlpackage/` | ML Program, fixed batch 16, fp16 weights, macOS 15+ |
| `manifest.json` | I/O contract, `logit_scale`, sha256, conversion versions, gate results |
| `golden.npz` | 32 real plant crops (preprocessed inputs) + torch fp32 reference embeddings, for a self-check after download |

## I/O contract

- **Input** `images`: `float32 [16, 3, 224, 224]`, preprocessed exactly like
  `bioclip.predict.preprocess_img` (resize to 224×224 without crop, CLIP
  mean/std normalization). Pad a short final batch (e.g. repeat the last
  image) and drop the padded outputs.
- **Output** `emb`: `[16, 768]`, **not normalized** — L2-normalize before use.
- `logit_scale` = 100.0 (from the source model), for
  `softmax(logit_scale · img_emb · txt_embᵀ)`.

```python
import coremltools as ct, numpy as np
m = ct.models.MLModel("encoder_b16.mlpackage", compute_units=ct.ComputeUnit.ALL)
emb = m.predict({"images": batch})["emb"]          # batch: float32 [16,3,224,224]
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
```

The first load compiles the package (~10 s); cache the compiled
`.mlmodelc` (`MLModel.get_compiled_model_path()`) and load it with
`ct.models.CompiledMLModel` afterwards. Inference works with numpy 2.x;
the numpy < 2.4 limit of coremltools 9 applies to conversion only.

## Fidelity

Gate: per-image cosine vs. torch fp32 CPU, mean ≥ 0.999 and min ≥ 0.998.

| Set | Compute units | cos mean / min | Throughput (Mac mini M4) |
|---|---|---|---|
| `golden.npz` (32 crops) | ALL (incl. Neural Engine) | 0.999712 / 0.999259 | 32 img/s (encoder only) |
| `golden.npz` (32 crops) | CPU_AND_GPU | 0.999996 / 0.999978 | 21 img/s |
| 559 library photos, end to end vs. Linux torch | ALL | 0.999712 / 0.998195 | — |

End to end on the 559-photo run (BioCLIP 2 zero-shot over the ~867k-species
label table):

- kingdom agreement 99.8 %;
- species top-1 within the torch top-5: 100 %;
- species top-1 agreement: 95.8 %.

The remaining top-1 flips are near-ties between closely related species,
whose text embeddings sit very close together. With `CPU_AND_GPU`
compute units the agreement is ~99 % at about two thirds of the speed.

For comparison, torch on the M4's CPU runs at about 8 img/s (encode +
classification), so this is roughly 3× faster than any CPU path,
including Docker on a Mac.

## Conversion

`open_clip.create_model_from_pretrained("hf-hub:imageomics/bioclip-2")`,
visual tower only, traced with the MHA fast path disabled, converted to an
ML Program with coremltools 9.0 (torch 2.7.0, open_clip 4.2.2,
numpy 1.26.4). Script:
[`scripts/export_teacher_coreml.py`](https://github.com/cxcxcxcx/botanisense-for-immich/blob/main/scripts/export_teacher_coreml.py);
verification: `scripts/verify_coreml_teacher.py`.

## License and attribution

MIT, same as the source model; see `LICENSE`. All credit for the model
goes to the BioCLIP 2 authors at the Imageomics Institute — this is only a
format conversion. Please cite their work:

```bibtex
@software{Gu_BioCLIP_2_model,
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila M Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  license = {MIT},
  title = {{BioCLIP 2}},
  url = {https://huggingface.co/imageomics/bioclip-2},
  version = {1.0.0},
  doi = {10.57967/hf/5765},
  publisher = {Hugging Face},
  year = {2025}
}

@inproceedings{NEURIPS2025_94da80cb,
  author = {Gu, Jianyang and Stevens, Sam and Campolongo, Elizabeth and Thompson, Matthew and Zhang, Net and Wu, Jiaman and Kopanev, Andrei and Mai, Zheda and White, Alexander and Balhoff, James and Dahdul, Wasila and Rubenstein, Daniel and Lapp, Hilmar and Berger-Wolf, Tanya and Chao, Wei-Lun (Harry) and Su, Yu},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning},
  volume = {38},
  pages = {102778--102811},
  year = {2025}
}
```
