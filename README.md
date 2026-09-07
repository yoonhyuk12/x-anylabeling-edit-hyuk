# x-anylabeling-edit-hyuk

[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) v4.0.6 기반의 수정본입니다.
네트워크 폴더 로딩 개선, 이미지·라벨 미리 읽기, Windows 긴 경로 처리를 포함합니다.

## Windows에서 실행

1. Python 3.12를 설치합니다. 설치 시 Python Launcher도 포함해야 합니다.
2. 이 저장소를 내려받아 압축을 풀거나 아래 명령으로 복제합니다.
3. `X-AnyLabeling 실행.bat`를 더블클릭합니다. 처음에는 인터넷 연결을 통해 CPU용 환경과 패키지를 자동 설치하며, 이후에는 앱을 실행합니다.

```powershell
git clone https://github.com/yoonhyuk12/x-anylabeling-edit-hyuk.git
cd x-anylabeling-edit-hyuk
& '.\X-AnyLabeling 실행.bat'
```

Git으로 설치했다면 앱을 종료한 뒤 `git pull --ff-only`로 업데이트합니다.
패키지 의존성이 변경된 경우 `.\.venv-cpu\Scripts\python.exe -m uv pip install -e ".[cpu]"`를 다시 실행합니다.
가상환경과 AI 모델 가중치는 저장소에 포함하지 않습니다. GPU 설치 방법은 아래 원본 문서를 참고하세요.

## 여러 사람이 사용할 때

각자 PC에 설치한 뒤 접근 권한이 있는 공유 폴더를 열어 사용할 수 있습니다.
작업 자동 배정이나 동시 편집 잠금은 없으므로 **담당 이미지 또는 하위 폴더를 나눠 작업하세요.**
같은 이미지의 라벨을 동시에 수정하면 나중에 저장한 내용으로 덮어써질 수 있습니다.

설정의 `Fast Folder Loading`은 폴더를 열 때 검토 상태를 필요할 때만 읽습니다.
`Images To Read Ahead`는 미리 읽을 이미지 수이며 기본값은 5, 0으로 설정하면 비활성화됩니다.

원본 프로젝트의 GPL-3.0 라이선스와 저작자 표기를 유지합니다. 아래는 원본 프로젝트 안내입니다.

---

<div align="center">
  <p>
    <a href="https://github.com/CVHub520/X-AnyLabeling/" target="_blank">
      <img alt="X-AnyLabeling" height="200px" src="anylabeling/resources/images/logo.png"></a>
  </p>

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-GPL%20v3-blue.svg"></a>
    <a href="https://github.com/CVHub520/X-AnyLabeling/releases/latest"><img src="https://img.shields.io/github/v/release/CVHub520/X-AnyLabeling?color=ffa"></a>
    <a href="https://pypi.org/project/x-anylabeling-cvhub/"><img src="https://img.shields.io/pypi/v/x-anylabeling-cvhub?logo=pypi&logoColor=white"></a>
    <a href="./pyproject.toml"><img src="https://img.shields.io/badge/python-3.11+-aff.svg"></a>
    <a href="https://github.com/CVHub520/X-AnyLabeling/releases/latest"><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/CVHub520/X-AnyLabeling/releases"><img src="https://img.shields.io/github/downloads/CVHub520/X-AnyLabeling/total?label=downloads"></a>
    <a href="https://modelscope.cn/collections/X-AnyLabeling-7b0e1798bcda43"><img src="https://img.shields.io/badge/modelscope-X--AnyLabeling-6750FF?link=https%3A%2F%2Fmodelscope.cn%2Fcollections%2FX-AnyLabeling-7b0e1798bcda43"></a>
</p>

<p align="center">
  <a href="https://mp.weixin.qq.com/s/Amccih1cM-B3hEfaf6koyw" target="_blank">
    <img src="https://github.com/user-attachments/assets/0bb9e00b-060b-4ce9-91c4-3666e29e4bf2" alt="YOLO Vision 2026" width="100%" />
  </a>
</p>

<img src="https://github.com/user-attachments/assets/aa819dae-e38c-4b1c-a4a7-53a873d870e3" alt="X-AnyLabeling interface" width="100%" />

## 🥳 What's New

- `2026-08-19`: Add support for [image tagging](https://xanylabeling.com/docs/x-anylabeling/user_guide#37-image-tags), with tag creation, editing, reordering, and batch deletion.
- `2026-08-12`: Add support for [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg) instance segmentation models.
- `2026-08-08`: Add support for the [RT-DETRv2-OBB](https://xanylabeling.com/examples/detection/obb) rotated object detection model.
- `2026-08-08`: Add the [Magic Wand tool](https://xanylabeling.com/docs/x-anylabeling/user_guide#21-creating-shapes) for quickly creating polygons from contiguous color regions.
- `2026-08-05`: Release [X-AnyLabeling v4.0.0](https://mp.weixin.qq.com/s/RUb8ge-_7br0YeIImuK6YQ).
- For more details, please refer to the [CHANGELOG](./CHANGELOG.md)

## Introduction

**X-AnyLabeling** is a lightweight, efficient, and unified cross-platform desktop application for AI-assisted annotation of text, image, video, and multimodal data. It combines versatile built-in tools, automated labeling workflows, state-of-the-art deep learning models, and flexible multi-format import and export. For remote inference, [X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server) provides a lightweight, extensible backend for connecting custom models and compute resources.

## Key Features

<img src="https://github.com/user-attachments/assets/2925bc88-e22b-4e81-873c-45fd85164f6b" width="100%" />

* Unified support for annotating and processing text, image, video, and multimodal data.
* Covers tasks such as image classification, object detection, instance segmentation, pose estimation, oriented object detection, multi-object tracking, optical character recognition, lane annotation, image captioning, visual question answering, and document parsing.
* Provides polygons, rectangles, cuboids, rotated boxes, quadrilaterals, circles, lines, polylines, points, masks, and task-specific tools for text detection, text recognition, and KIE.
* Integrates a wide range of state-of-the-art deep learning models for AI-assisted annotation, automated labeling, and batch dataset prediction.
* Supports both local and remote inference through engines and serving frameworks such as `ONNX Runtime`, `TensorRT`, `OpenCV DNN`, `vLLM`, and `SGLang`.
* Supports importing and exporting formats such as `COCO`, `VOC`, `YOLO`, `DOTA`, `MOT`, `MASK`, `PPOCR`, `MMGD`, `VLM-R1`, and `ShareGPT`.
* Runs on Windows, Linux, and macOS, with interfaces available in English, Simplified Chinese, Japanese, and Korean.
* Supports custom model integration, flexible extension, and secondary development.

## Model library

| **Task Category** | **Supported Models** |
| :--- | :--- |
| 🖼️ Image Classification | YOLOv5-Cls, YOLOv8-Cls, YOLO11-Cls, InternImage, PULC |
| 🎯 Object Detection | YOLOv5/6/7/8/9/10, YOLO11/12/26, YOLOX, YOLO-NAS, D-FINE, DAMO-YOLO, Gold_YOLO, RT-DETR, RF-DETR, DEIMv2 |
| 🖌️ Instance Segmentation | YOLOv5-Seg, YOLOv8-Seg, YOLO11-Seg, YOLO26-Seg, Hyper-YOLO-Seg, RF-DETR-Seg, D-FINE-seg |
| 🏃 Pose Estimation | YOLOv8-Pose, YOLO11-Pose, YOLO26-Pose, DWPose, RTMO |
| 😀 Face Estimation | SCRFD, YOLOv6Lite-Face |
| 👣 Tracking | TrackTrack, Bot-SORT, ByteTrack, SAM2/3-Video |
| 🔄 Rotated Object Detection | YOLOv5-Obb, YOLOv8-Obb, YOLO11-Obb, YOLO26-Obb, RT-DETRv2-OBB |
| 📏 Depth Estimation | Depth Anything |
| 🧩 Segment Anything | SAM 1/2/3, SAM-HQ, SAM-Med2D, EdgeSAM, EfficientViT-SAM, MobileSAM |
| ✂️ Image Matting | RMBG 1.4/2.0 |
| 💡 Proposal | UPN |
| 🏷️ Tagging | RAM, RAM++ |
| 📄 OCR | PP-OCRv4, PP-OCRv5, PP-OCRv6 |
| 🧾 Layout Analysis | PP-DocLayoutV3 |
| 📑 Document Parsing | PaddleOCR-VL, PaddleOCR-VL-1.6 |
| 🗣️ Vision Foundation Models | Rex-Omni, Florence2 |
| 👁️ Vision Language Models | Qwen3-VL, Gemini, ChatGPT, GLM |
| 🛣️ Lane Detection | CLRNet |
| 🔢 Object Counting | CountGD, GeCO, GeCo2 |
| 📍 Grounding | Grounding DINO, YOLO-World, YOLOE, SAM 3, LocateAnything |
| 📚 Other | 👉 [model_zoo](./docs/en/model_zoo.md) 👈 |

## Docs

0. [Remote Inference Service](https://github.com/CVHub520/X-AnyLabeling-Server)
1. [Installation & Quickstart](./docs/en/get_started.md)
2. [Usage](./docs/en/user_guide.md)
3. [Command Line Interface](./docs/en/cli.md)
4. [Customize a model](./docs/en/custom_model.md)
5. [Chatbot](./docs/en/chatbot.md)
6. [VQA](./docs/en/vqa.md)
7. [Image Classifier](./docs/en/image_classifier.md)
8. [Video Classifier](./docs/en/video_classifier.md)
9. [Document Parsing and Intelligent Text Recognition](./docs/en/paddle_ocr.md)

## Examples

- [Classification](./examples/classification/)
  - [Image-Level](./examples/classification/image-level/README.md)
  - [Shape-Level](./examples/classification/shape-level/README.md)
- [Detection](./examples/detection/)
  - [HBB Object Detection](./examples/detection/hbb/README.md)
  - [OBB Object Detection](./examples/detection/obb/README.md)
- [Segmentation](./examples/segmentation/README.md)
  - [Instance Segmentation](./examples/segmentation/instance_segmentation/)
  - [Binary Semantic Segmentation](./examples/segmentation/binary_semantic_segmentation/)
  - [Multiclass Semantic Segmentation](./examples/segmentation/multiclass_semantic_segmentation/)
- [Description](./examples/description/)
  - [Tagging](./examples/description/tagging/README.md)
  - [Captioning](./examples/description/captioning/README.md)
- [Estimation](./examples/estimation/)
  - [Face Estimation](./examples/estimation/face_estimation/README.md)
  - [Pose Estimation](./examples/estimation/pose_estimation/README.md)
  - [Depth Estimation](./examples/estimation/depth_estimation/README.md)
- [OCR](./examples/optical_character_recognition/)
  - [Text Recognition](./examples/optical_character_recognition/text_recognition/)
  - [Key Information Extraction](./examples/optical_character_recognition/key_information_extraction/README.md)
- [MOT](./examples/multiple_object_tracking/README.md)
  - [Tracking by HBB Object Detection](./examples/multiple_object_tracking/README.md)
  - [Tracking by OBB Object Detection](./examples/multiple_object_tracking/README.md)
  - [Tracking by Instance Segmentation](./examples/multiple_object_tracking/README.md)
  - [Tracking by Pose Estimation](./examples/multiple_object_tracking/README.md)
- [iVOS](./examples/interactive_video_object_segmentation)
  - [SAM2-Video](./examples/interactive_video_object_segmentation/sam2/README.md)
  - [SAM3-Video](./examples/interactive_video_object_segmentation/sam3/README.md)
- [Matting](./examples/matting/)
  - [Image Matting](./examples/matting/image_matting/README.md)
- [Vision-Language](./examples/vision_language/)
  - [Rex-Omni](./examples/vision_language/rexomni/README.md)
  - [Florence 2](./examples/vision_language/florence2/README.md)
- [Counting](./examples/counting/)
  - [GeCo](./examples/counting/geco/README.md)
  - [GeCo2](./examples/counting/geco2/README.md)
- [Grounding](./examples/grounding/)
  - [YOLOE](./examples/grounding/yoloe/README.md)
  - [SAM 3](./examples/grounding/sam3/README.md)
  - [LocateAnything](./examples/grounding/locateanything/README.md)
- [Training](./examples/training/)
  - [Ultralytics](./examples/training/ultralytics/README.md)

## Contribute

We believe in open collaboration! **X‑AnyLabeling** continues to grow with the support of the community. Whether you're fixing bugs, improving documentation, or adding new features, your contributions make a real impact.

To get started, please read our [Contributing Guide](./CONTRIBUTING.md) and make sure to agree to the [Contributor License Agreement (CLA)](./CLA.md) before submitting a pull request.

If you find this project helpful, please consider giving it a ⭐️ star! Have questions or suggestions? Open an [issue](https://github.com/CVHub520/X-AnyLabeling/issues) or email us at cv_hub@163.com.

A huge thank you 🙏 to everyone helping to make X‑AnyLabeling better.

## License

This project is licensed under the [GNU General Public License v3.0](./LICENSE). You may use, modify, and redistribute the software, including for commercial purposes, provided that you comply with the terms of the license.

## Sponsor

X-AnyLabeling is an actively maintained open-source project. Your sponsorship helps support feature development, model integration, documentation, and community support.

<a href="https://xanylabeling.com/sponsor">
  <img src="https://github.com/user-attachments/assets/893151ad-d6b2-4846-882a-ef5376471c99" alt="Sponsor the X-AnyLabeling project" width="100%" />
</a>

Click the image above to visit the sponsorship page.

## Acknowledgement

I extend my heartfelt thanks to the developers and contributors of [AnyLabeling](https://github.com/vietanhdev/anylabeling), [LabelMe](https://github.com/wkentaro/labelme), [LabelImg](https://github.com/tzutalin/labelImg), [roLabelImg](https://github.com/cgvict/roLabelImg), [PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel) and [CVAT](https://github.com/opencv/cvat), whose work has been crucial to the success of this project.

## Citing

If you use this software in your research, please cite it as below:

```
@misc{X-AnyLabeling,
  year = {2023},
  author = {Wei Wang},
  publisher = {Github},
  organization = {CVHub},
  journal = {Github repository},
  title = {X-AnyLabeling: A Unified Desktop Platform for AI-Assisted Data Annotation},
  howpublished = {\url{https://github.com/CVHub520/X-AnyLabeling}}
}
```

<div align="center"><a href="#top">🔝 Back to Top</a></div>
