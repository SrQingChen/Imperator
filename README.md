<div align="center">

<img src="assets/icon.svg" width="110" alt="英白拉多 Imperator">

# 英白拉多 · Imperator

**把混在一起的声音，分而治之。**

视频丢进来，BGM 归 BGM，每个说话人一条独立音轨——**两人同时说话的重叠段也能正确归属**。

[![License GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue?style=flat-square&logo=gnu)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](requirements.txt)
[![FastAPI](https://img.shields.io/badge/Web-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Platform Windows](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-0078D6?style=flat-square&logo=windows)](#)
[![Release 1.0.0](https://img.shields.io/badge/Release-1.0.0-e8c97a?style=flat-square)](#)
[![Models Mel-RoFormer · CAM++ · MossFormer2](https://img.shields.io/badge/%E6%A8%A1%E5%9E%8B-MelRoFormer%20%C2%B7%20CAM%2B%2B%20%C2%B7%20MossFormer2-c9a54e?style=flat-square)](#技术架构)
[![镜像自动探测](https://img.shields.io/badge/%E5%9B%BD%E5%86%85-%E9%95%9C%E5%83%8F%E8%87%AA%E5%8A%A8%E5%88%87%E6%8D%A2-FF6A00?style=flat-square)](#环境与模型自动管理)

**分离Separation，正是岁主的权能。** 一个自托管、可配置、可审计的智能音轨分离器。

</div>

---

## 为什么需要它

你大概经历过这些时刻：

- 🎬 **访谈素材里 BGM 太吵**：UVR / Demucs 能去掉伴奏，但输出的人声轨里，**三个人说话仍然糊成一团**——因为 "vocals" 是一个语义类别，不是一个个人。
- 🎙️ **多人播客要逐人精修**：想在后期单独调整某个人的音量、删掉某人的一句口误，却发现根本没有「某人的一条轨」。
- 🤖 **AI 转写分不清谁在说**：ASR 输出的对话没有可靠的主人，尤其是两人**抢话重叠**的那几秒。
- 📦 **现有工具链拼不起来**：pyannote 给时间线不给音轨，AudioShake 能拆但是商业黑盒，UVR 不懂说话人——把这几件事拼好，本来要一周工程活。

**英白拉多把这些全部装进一个本地 Web 应用**：导入视频 → 自动抽音轨 → 剥离 BGM → 按说话人分轨 → 重叠段分别归属 → 输出 N 条 WAV。

## ✨ 核心特性

### 分离能力

| 能力 | 说明 |
|---|---|
| 🎵 **BGM / 人声分离** | Mel-RoFormer（当前最强开源人声分离模型之一），伴奏完整保留为独立轨 |
| 🗣️ **按说话人分轨** | CAM++ 声纹滑窗 + 可配置阈值聚类：**"两个人声多像算同一个人"由你定义** |
| ⚡ **重叠说话分离** | 三信号触发 MossFormer2 双流盲分离 + 声纹匹配归属——**两人同时说话，各回各轨** |
| 🔁 **偶发排除** | 惊呼 / 咳嗽 / 短暂变声不再凭空变成"第四说话人"：并入最相似的人，或进入未归属轨，每个决定可审计 |
| 📐 **精确输出契约** | 每条轨与源**逐样本等长**、无人声处**精确静音**、**全部音轨相加 = 原始声音**（残差轨显式兜底，交付前自动断言） |

### 工程体验

| 特性 | 说明 |
|---|---|
| 🖥️ **优雅 Web UI** | FastAPI 后端 + 单页深色前端：任务队列、九阶段进度、终端风实时日志、多轨同步试听（solo/mute）、说话人时间线 |
| 🧰 **依赖/模型全自动** | 缺什么补什么：依赖体检一键安装、模型按需下载、断点续传、体积校验 |
| 🇨🇳 **镜像智能切换** | 启动即探测网络（境外连通性 + 出口 IP 双重判定），国内自动走清华 pip 源 / hf-mirror / ghproxy 链，ModelScope 直连 |
| 📊 **性能监控** | CPU / 内存 / GPU 利用率 / 显存实时曲线 + 每任务 RTF 与分阶段耗时 |
| 🧠 **声纹库** | 说话人存库命名后，跨任务同一人自动同名输出；本地存储不上传 |
| 🔍 **QC 质检报告** | Σ 还原偏差、残差能量比、每轨占比、低置信度清单、审计日志（偶发排除的每个决定带分项分数） |
| ✏️ **人工修正回路** | 时间线上点选任意段落改派说话人 → **秒级局部重渲染**（不重跑模型），全程留痕 |

## 🚀 快速开始

```bash
git clone https://github.com/SrQingChen/Imperator.git
cd Imperator
python -m venv .venv
.venv\Scripts\activate          # Windows（Linux/macOS: source .venv/bin/activate）
pip install -r requirements.txt # 国内网络会自动可用镜像，或自行加 -i 清华源

python main.py                  # 打开 http://127.0.0.1:8765
```

首次使用：进入 **模型与环境** 页一键下载缺失依赖与模型（约 2–4 GB，走镜像）；然后拖一个视频进 **分离任务** 页即可。也可用脚本离线预下载：

```bash
python scripts/fetch_models.py --list
python scripts/fetch_models.py --all
```

> GPU 可选：有 NVIDIA 显卡自动用 CUDA（标准档更快）；纯 CPU 也能完整运行（建议选"轻量"档）。torch 的 CUDA 版只在 PyTorch 官方源分发，安装器已单独处理。

## 🔄 它是如何工作的

```
视频/音频 ──ffmpeg──▶ 44.1kHz 主轴音频
                        │
        ┌───────────────▼────────────────┐
        │ Mel-RoFormer 伴奏分离            │  → bgm.wav
        └───────────────┬────────────────┘
                        │ vocals
                        ▼
        16kHz 分析支线：Silero VAD → CAM++ 声纹滑窗
                        │
                        ▼
        凝聚聚类（阈值 τ 可调）──▶ 偶发排除（比例/自动评分）──▶ 声纹库命名
                        │
                        ▼
        重叠检测（歧义窗/切人边界/自洽性）──▶ MossFormer2 双流分离
                        │  双流分别命中两个说话人 → 软比率掩码（m_a + m_b ≡ 1）
                        ▼
        能量账本重建：speaker_i = m_i · vocals
        other = (1−Σm_i) · vocals      bgm = instrumental
        residual = master − vocals − instrumental
                        │
                        ▼
        Σ speakers + other + bgm + residual ≡ master   （逐样本浮点恒等，交付前断言）
```

**为什么"相加精确还原"值得强调**：真实混音并非线性叠加，任何分离模型都有残差。英白拉多把残差显式做成一条轨道、把掩码互补性写进数学（相邻段淡变共用同一升余弦分区形状，补集掩码 `1−Σm_i` 不截断），于是"**所有轨加起来等于原声**"成为用户可验证、代码可断言的契约，而不是营销话术。

## ⚙️ 高级配置（节选）

- **合并置信度 τ**：两段人声余弦相似度 ≥ τ 即视为同一人（默认 0.62，附校准脚本 `scripts/calibrate.py`）
- **偶发排除**：
  - *比例模式*（默认）：某"说话人"总时长占比低于 4%（可调）→ 并入最相似的人
  - *自动模式*：六因子综合评分（相似度距离 / 与次候选差距 / 时长短暂度 / 上下文连续性 / 能量常态性 / 语音性）动态判定并入 / 未归属 / 待复核，分项分数全部入审计
  - *持续证据原则*：不足"≥3 段且累计 ≥5 秒"的证据，不新建说话人——**宁可 unknown，不强行合并**
- **重叠参数**：触发 margin、上下文长度
- **输出**：float32 / 24-bit / 16-bit WAV，44.1k / 48k / 跟随源
- **伴奏档位**：标准 Mel-RoFormer / 快速 BS-RoFormer / 轻量 MDX（CPU 友好）/ 跳过（纯人声）

## 🧭 能力边界（诚实声明）

- 单声道、同性别、同音色、无任何先验的多人**同时**说话，是物理欠定问题——英白拉多会给出置信度、未归属轨与残差，**不会假装完美**。
- 3 人以上同时重叠为实验性（归属主导说话人并打 `overlap_multi` 标记）。
- BGM 含歌声时，原唱进入「未归属人声」轨，不会污染说话人轨。
- 主导型重叠（两人音量悬殊）的检出率有限，UI 中以低置信度标示。
- 取消任务在阶段边界生效（大模型推理不可中断），UI 有如实提示。

## 🧱 技术架构

| 层 | 选型 | 许可 |
|---|---|---|
| 伴奏/人声 (MSS) | [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) · Mel-RoFormer / BS-RoFormer / MDX | MIT |
| 语音检测 (VAD) | [silero-vad](https://github.com/snakers4/silero-vad) | MIT |
| 声纹与说话人 | [FunASR](https://github.com/modelscope/FunASR) · CAM++ | MIT |
| 重叠双流分离 | [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) · MossFormer2_SS_16K | Apache-2.0 |
| 解码 | ffmpeg（系统或 imageio-ffmpeg 兜底） | — |
| Web | FastAPI + uvicorn + 原生 JS 单页前端 | MIT 系 |

代码以 **GPL-3.0** 发布（与上述 MIT / Apache-2.0 依赖单向兼容）。模型权重不随本仓库分发，运行时从各自官方渠道（ModelScope / HuggingFace，国内自动镜像）下载，版权归各自作者，详见 [NOTICE.md](NOTICE.md)。

## 🗺️ Roadmap

- [ ] v1.1：声纹库跨任务自动聚类合并
- [ ] v2：LExt 式目标说话人提取（TSE）替代盲分离（重叠质量再上一档）
- [ ] v2：已知歌曲参考消除（指纹识别 → 对齐 → 自适应滤波减法）
- [ ] v2：歌声 / 说话判别与歌声独立轨
- [ ] 评估：3+ 人重叠、多通道阵列、音视频口型先验

## 📜 命名由来与致敬

**英白拉多**是拉丁语 **Imperator** 的音译——古罗马共和国时期，士兵在重大胜利后向统帅高呼的尊号；帝国时期成为皇帝的头衔，英语 "emperor" 即由此演变，词根 *imperāre* 意为「命令」。

项目名灵感来自游戏《鸣潮》中黎那汐塔的岁主「英白拉多」——掌「分离」之权能。图标中的月桂花环致敬罗马的荣耀意象，中央波形一分为二，即「分离」本身。

> **免责声明**：本项目仅为个人兴趣开发，仅受上述作品的概念启发而命名，与库洛游戏（Kuro Games）及相关方**没有任何关联、合作、授权或背书**；《鸣潮》及其角色名称、设定之版权归其权利人所有。

## 👤 作者

**SrQingChen** · GPL-3.0 · 欢迎议题与 PR

<div align="center"><sub>把混在一起的声音，分而治之。</sub></div>
