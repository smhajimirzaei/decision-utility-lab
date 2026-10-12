# Third-party notices

## Meta-World

The recorded images and videos were rendered in [Meta-World](https://github.com/Farama-Foundation/Metaworld). Meta-World is copyright (c) 2019 Meta-World Team and distributed under the MIT license. The full notice is included in [`licenses/MetaWorld-MIT.txt`](licenses/MetaWorld-MIT.txt). Simulator assets retain their upstream rights; project licensing does not replace them.

## LeRobot / SmolVLA

The frozen policy used [LeRobot](https://github.com/huggingface/lerobot) and the [`lerobot/smolvla_metaworld`](https://huggingface.co/lerobot/smolvla_metaworld) checkpoint. The pinned source/model manifests are in `configs/smolvla_metaworld_manifest.json`. Upstream licensing is Apache-2.0. This repository does not redistribute those source trees or model weights.

## JEPA-WMs

The dynamics adapter uses [JEPA-WMs](https://github.com/facebookresearch/jepa-wms), with revisions recorded in `configs/jepa_metaworld_manifest.json`. The upstream project is licensed CC BY-NC 4.0, with per-file notices. Its noncommercial terms remain applicable to upstream material and weights; a license for this project's own code does not override them. Upstream source trees, pretrained weights, and raw latent archives are not redistributed here.

## MuJoCo and other dependencies

[MuJoCo](https://github.com/google-deepmind/mujoco) and optional research dependencies are installed separately under their respective licenses. Dependency declarations are not permission to redistribute upstream model weights or simulator assets under a different license.
