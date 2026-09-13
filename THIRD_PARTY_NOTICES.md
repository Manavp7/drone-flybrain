# Third-party notices

The project software is licensed under [Apache-2.0](LICENSE), copyright 2026
Manavp7. Third-party source, media and separately acquired artifacts retain
their own terms. This repository does not include pretrained checkpoints,
downloaded source videos, Python environments or dependency distributions.

## YOLOX

The project's `perception/detector.py` implements the input preprocessing,
grid decoding and COCO class contracts documented by
[Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).
The project adds its own validation, bounds, OpenCV adapter, tiling and
tracking integration; it does not contain a vendored YOLOX package.

Copyright (c) 2021-2022 Megvii Inc. All rights reserved.

YOLOX source is licensed under Apache-2.0. The unmodified upstream license
and attribution notice are preserved in
[LICENSES/YOLOX-Apache-2.0.txt](LICENSES/YOLOX-Apache-2.0.txt).

Primary references:

- [Upstream license](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE)
- [ONNX inference example](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py)
- [Preprocessing](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py)
- [Decoding utilities](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py)
- [Class names](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/datasets/coco_classes.py)
- [Official model release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)

Model files are acquired separately from the official release. Their download
URLs and integrity hashes identify artifacts; they do not create a new license
grant for those artifacts.

## Flyvis

[Flyvis](https://github.com/TuragaLab/flyvis) is a separately installed research
dependency. The recorded experiments use Flyvis 1.2.0 with source revision
`92b3845cc426dd309a1a0e1b3890156c42e14021`.

Copyright (c) 2023 Janne K. Lappalainen, Fabian D. Tschopp, Mason McGill,
Jakob H. Macke, Srinivas C. Turaga.

Flyvis source is licensed under MIT. The unchanged
[pinned upstream license](https://github.com/TuragaLab/flyvis/blob/92b3845cc426dd309a1a0e1b3890156c42e14021/license)
is included as [LICENSES/Flyvis-MIT.txt](LICENSES/Flyvis-MIT.txt). This copy
records the dependency's source notice; it does not mean that Flyvis itself is
bundled here.

The pretrained research archive used in the local experiments contained no
separate checkpoint license statement. Its model files are excluded from this
repository. The MIT source license is not presented as evidence of checkpoint
redistribution rights. Acquire any required checkpoint from upstream and
review the terms supplied there. The project's small fitted ridge readout is
a separate project artifact, not a copy of the pretrained Flyvis checkpoint.

Primary references:

- [Research publication](https://doi.org/10.1038/s41586-024-07939-3)
- [Official custom-stimulus workflow](https://turagalab.github.io/flyvis/examples/07_flyvision_providing_custom_stimuli/)
- [Pinned upstream model downloader](https://github.com/TuragaLab/flyvis/blob/92b3845cc426dd309a1a0e1b3890156c42e14021/flyvis_cli/download_pretrained_models.py)

## MuJoCo and other runtime dependencies

[MuJoCo 3.2.7](https://pypi.org/project/mujoco/3.2.7/) is separately installed
and provides the rigid-body simulator. Its source is Apache-2.0, with
additional notices for its own dependencies. Refer to the
[official source license](https://github.com/google-deepmind/mujoco/blob/3.2.7/LICENSE)
and to `LICENSE` and `LICENSES_THIRD_PARTY.md` in the installed distribution.
No MuJoCo binary, Python environment or dependency source package is bundled
in this repository.

Other packages listed in the requirements files are also installed separately
and retain their own licenses. Their inclusion as dependency names does not
relicense their code or assets under this project's license.

## Demonstration media

The photograph and derived demonstration files use the CC BY 3.0 Sabana
Grande footage by Vicente Quintero / QuinteroP. These media files have a
separate license from the software. See [assets/ATTRIBUTION.md](assets/ATTRIBUTION.md)
for the exact included files, source, author, license and modifications. The
unmodified standard license text is in
[LICENSES/CC-BY-3.0.txt](LICENSES/CC-BY-3.0.txt).

The original Wikimedia and YouTube video inputs are not included. Mentioning
an evaluated video or a model does not imply endorsement by its creators.

## License-text sources

License files are verbatim copies, without project-specific additions inside
the standard text:

- `LICENSE`: [Apache Software Foundation's Apache-2.0 text](https://www.apache.org/licenses/LICENSE-2.0.txt).
- `LICENSES/YOLOX-Apache-2.0.txt`: [YOLOX's upstream license and Megvii notice](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/LICENSE).
- `LICENSES/Flyvis-MIT.txt`: [Flyvis's pinned MIT text and contributor notice](https://raw.githubusercontent.com/TuragaLab/flyvis/92b3845cc426dd309a1a0e1b3890156c42e14021/license).
- `LICENSES/CC-BY-3.0.txt`: [SPDX's standard CC BY 3.0 text](https://raw.githubusercontent.com/spdx/license-list-data/main/text/CC-BY-3.0.txt), corresponding to the [Creative Commons legal code](https://creativecommons.org/licenses/by/3.0/legalcode).
