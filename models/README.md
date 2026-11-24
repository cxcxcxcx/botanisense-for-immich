# obj6 RTMDet detector

`obj6_rtmdet_ep97.onnx` is the default detection backend of BotaniSense
for Immich (`YOLO_BACKEND=rtmdet`, loaded by `yolo_rtmdet.py`). It finds
the regions the classifier then looks at.

| | |
|---|---|
| Architecture | RTMDet-Ins-tiny ([mmdetection](https://github.com/open-mmlab/mmdetection), Apache-2.0) |
| Classes | `plant`, `person`, `hand`, `flower`, `tree` |
| Input | 960 × 960 RGB, letterboxed |
| Format | ONNX, fp32, 20.5 MB, 6.47 M parameters |
| sha256 | `911c800ce98391b6ed743921ffe10a0d37a58d4b1a909198709a0fd472e058f3` |
| Box AP50 | .556 on the 81-image validation split |

**Training data.** Boxes and masks auto-labeled on the author's own
photo library, then reviewed by hand. The images are not distributed.

Export, parity check and measured trade-offs:
[docs/detector_backends.md](../docs/detector_backends.md).

## License

The weights in this directory are licensed under
[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0) — full text in [LICENSE](LICENSE). © 2026 Xing Chen.

Non-commercial use, sharing and adaptation are welcome with attribution.
For commercial use, open an issue or contact the author.

The code in this repository (including the code that loads and runs
this model) is MIT; see the top-level [LICENSE](../LICENSE).
