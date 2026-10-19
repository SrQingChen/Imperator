# NOTICE · 第三方组件与模型权重

英白拉多 Imperator 的代码以 GPL-3.0 发布。本项目**不随仓库分发任何模型权重**；权重在首次使用时从各自官方渠道下载（国内网络自动走镜像），其版权与许可归各自作者所有。

## Python 组件

| 组件 | 许可 | 用途 |
|---|---|---|
| [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | MIT | 伴奏/人声分离（RoFormer / MDX / VR / Demucs 推理） |
| [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) | Apache-2.0 | MossFormer2_SS_16K 双说话人盲分离 |
| [FunASR](https://github.com/modelscope/FunASR) | MIT（代码） | CAM++ 说话人声纹 |
| [silero-vad](https://github.com/snakers4/silero-vad) | MIT | 语音活动检测 |
| FastAPI / uvicorn / Starlette | MIT / BSD 等 | Web 服务 |
| numpy / scipy / soundfile / psutil 等 | BSD 系 | 基础设施 |

## 模型权重（运行时下载）

| 权重 | 来源 | 说明 |
|---|---|---|
| Mel-Band RoFormer / BS-RoFormer / UVR-MDX 系列 | audio-separator 模型清单（HuggingFace / GitHub Releases） | 社区训练的 UVR 生态权重，多数未附明确许可证——请仅用于个人/研究用途，商用前自行向权重发布者确认 |
| MossFormer2_SS_16K | HuggingFace / ModelScope（ClearerVoice-Studio） | Apache-2.0 系生态发布 |
| CAM++（speech_campplus_sv_zh-cn_16k-common） | ModelScope（FunASR / 3D-Speaker 生态） | Apache-2.0 系生态发布 |
| Silero VAD | 随 silero-vad pip 包内置 | MIT |

## 数据与隐私

- 声纹属于生物特征数据：声纹库**仅保存在本机**（`data/voiceprints.json`），不上传、不遥测。
- 请只对你有权处理的音视频素材使用本工具；输出的分离结果仍受原素材版权约束。

## 命名

「英白拉多 / Imperator」仅为受《鸣潮》概念启发的命名，与 Kuro Games 无关联、未获背书；详见 README 免责声明。
