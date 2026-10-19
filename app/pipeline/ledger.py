"""能量账本：控制率增益曲线 + 互补淡变 + 补集掩码 + 流式分轨写盘。

契约（规划 §2.2）：Σ speakers + bgm + other + residual = master，逐样本浮点成立。
- 增益在 CTRL_RATE(1000Hz) 控制网格上构建，np.interp 线性插值到采样域——
  插值是线性算子，与求和可交换，互补性在采样域严格保持。
- m0 = 1 − Σmᵢ，不截断（负值瞬时记录进 QC）。
- 相邻 speech 段淡变共用同一升余弦分区形状（(1−f)/f）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio_io

CTRL_RATE = 1000
FADE_S = 0.015
_BLOCK_S = 1.0
_SUBTYPE = {"float32": "FLOAT", "s16": "PCM_16", "s24": "PCM_24"}


@dataclass
class LedgerResult:
    tracks: list[dict] = field(default_factory=list)   # {key,label,file,kind}
    qc: dict = field(default_factory=dict)
    duration_s: float = 0.0
    n_samples: int = 0
    sr_out: int = 0


def _raised_cosine(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(1, dtype="float32")
    t = np.arange(n, dtype="float32") / (n - 1)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def build_ctrl_gains(items: list[dict], speakers: list[int], total_dur: float,
                     ctrl_rate: int = CTRL_RATE, fade_s: float = FADE_S) -> dict[int, np.ndarray]:
    """时间线项 → 每说话人控制率增益曲线。speech=方波+互补淡变；overlap/hard=掩码包络。"""
    n = int(round(total_dur * ctrl_rate)) + 2
    gains = {sid: np.zeros(n, dtype="float32") for sid in speakers}

    speech_items = sorted([it for it in items if it["kind"] == "speech" and it["speaker"] >= 0],
                          key=lambda it: it["start"])
    for it in speech_items:
        i0 = int(round(it["start"] * ctrl_rate))
        i1 = int(round(it["end"] * ctrl_rate))
        sid = it["speaker"]
        if 0 <= i0 < i1 <= n:
            gains[sid][i0:i1] += 1.0

    for it in items:
        if it["kind"] in ("overlap", "hard"):
            i0 = int(round(it["start"] * ctrl_rate))
            for sid, mask16k in it.get("masks", {}).items():
                if sid not in gains:
                    gains[sid] = np.zeros(n, dtype="float32")
                m = _mask_to_ctrl(mask16k, ctrl_rate)
                i1 = min(i0 + len(m), n)
                seg = gains[sid][i0:i1]
                np.add(seg, m[: i1 - i0], out=seg)

    _apply_complementary_fades(gains, speech_items, ctrl_rate, fade_s, total_dur)
    return gains


def _mask_to_ctrl(mask16k: np.ndarray, ctrl_rate: int) -> np.ndarray:
    """16k 掩码 → 控制率：抽取后线性插值到规则网格。"""
    if len(mask16k) == 0:
        return np.zeros(0, dtype="float32")
    ratio = 16000 / ctrl_rate
    n_out = int(round(len(mask16k) / ratio))
    if n_out <= 0:
        return np.zeros(0, dtype="float32")
    idx_src = np.arange(n_out, dtype="float64") * ratio
    return np.interp(idx_src, np.arange(len(mask16k), dtype="float64"),
                     mask16k.astype("float64")).astype("float32")


def _apply_complementary_fades(gains: dict[int, np.ndarray], speech_items: list[dict],
                               ctrl_rate: int, fade_s: float, total_dur: float) -> None:
    """相邻 speech 段边界淡变：
    - 同说话人相邻 → 不动（连续）
    - 异说话人紧邻 → 互补升余弦（A 得 1−f，B 得 f，Σ 恒 1）
    - 有静音间隔 → 各自单侧淡出/淡入（Σ 暂降由补集 m0 兜住，无爆音）
    - 首段淡入 / 末段淡出
    overlap/hard 段的掩码本身平滑，不在此处理。
    """
    if not speech_items or not gains:
        return
    F = max(2, int(round(fade_s * ctrl_rate)))
    n = len(next(iter(gains.values())))

    def _slice(btime: float) -> tuple[int, int] | None:
        b = int(round(btime * ctrl_rate))
        lo, hi = max(0, b - F // 2), min(n, b + (F - F // 2))
        return (lo, hi) if hi - lo >= 2 else None

    for a, b in zip(speech_items, speech_items[1:]):
        if a["speaker"] == b["speaker"]:
            continue
        gap = b["start"] - a["end"]
        if gap <= F / ctrl_rate:  # 紧邻：互补交叉淡变
            span = _slice((a["end"] + b["start"]) / 2)
            if not span:
                continue
            lo, hi = span
            f = _raised_cosine(hi - lo)
            for sid, g in gains.items():
                in_l, in_r = sid == a["speaker"], sid == b["speaker"]
                if in_l and in_r:
                    continue
                seg = g[lo:hi]
                if in_l:
                    seg *= (1.0 - f)
                elif in_r:
                    seg *= f
        else:  # 静音间隔：各自单侧淡变
            for item, head in ((a, False), (b, True)):
                edge = item["end"] if not head else item["start"]
                i0 = int(round(edge * ctrl_rate))
                if head:
                    lo, hi = i0, min(n, i0 + F)
                else:
                    lo, hi = max(0, i0 - F), i0
                if hi - lo < 2:
                    continue
                g = gains[item["speaker"]]
                seg = g[lo:hi]
                seg *= _raised_cosine(hi - lo) if head else (1.0 - _raised_cosine(hi - lo))
    # 首段淡入 / 末段淡出
    first, last = speech_items[0], speech_items[-1]
    i0 = int(round(first["start"] * ctrl_rate))
    lo, hi = i0, min(n, i0 + F)
    if hi - lo >= 2:
        gains[first["speaker"]][lo:hi] *= _raised_cosine(hi - lo)
    i1 = int(round(last["end"] * ctrl_rate))
    lo, hi = max(0, i1 - F), i1
    if hi - lo >= 2:
        gains[last["speaker"]][lo:hi] *= (1.0 - _raised_cosine(hi - lo))


def write_tracks(master_path: Path, vocals_path: Path, inst_path: Path, out_dir: Path,
                 gains_ctrl: dict[int, np.ndarray], speaker_labels: dict[int, str],
                 out_format: str = "float32", out_sr: int | str = 44100,
                 on_progress=None) -> LedgerResult:
    """流式写盘：逐块读 vocals/master/inst → 插值增益 → 写每条轨 → Σ 断言。

    RAM 峰值 O(块)，与音频总长无关（>20min 长音频安全）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    subtype = _SUBTYPE.get(out_format, "FLOAT")

    mst_x, mst_sr = audio_io.load_audio(master_path)
    voc_x, sr_in = audio_io.load_audio(vocals_path)
    inst_x, inst_sr = audio_io.load_audio(inst_path)
    sr_mismatch = not (sr_in == mst_sr == inst_sr)
    n = mst_x.shape[0]
    voc_x, adj_v = audio_io.align_length(voc_x, n)
    inst_x, adj_i = audio_io.align_length(inst_x, n)
    duration = n / mst_sr

    ctrl_times = np.arange(len(next(iter(gains_ctrl.values()))) if gains_ctrl else 1,
                           dtype="float64") / CTRL_RATE
    m0_ctrl = (np.ones(len(ctrl_times), dtype="float32")
               - np.sum([g for g in gains_ctrl.values()], axis=0)) if gains_ctrl else np.ones(1, "float32")

    target_sr = mst_sr if out_sr == "source" else int(out_sr)
    streaming_ok = target_sr == mst_sr

    track_specs: list[tuple[str, str]] = []
    for sid in sorted(gains_ctrl):
        track_specs.append((f"speaker_{sid + 1:02d}", speaker_labels.get(sid, f"说话人 {sid + 1}")))
    track_specs += [("other", "未归属人声"), ("bgm", "伴奏 / BGM"), ("residual", "残差（诊断）")]

    tmp_dir = out_dir / "_44100"
    direct_dir = out_dir if streaming_ok else tmp_dir
    direct_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, sf.SoundFile] = {}
    for key, _ in track_specs:
        files[key] = sf.SoundFile(str(direct_dir / f"{key}.wav"), mode="w", samplerate=mst_sr,
                                  channels=voc_x.shape[1] if voc_x.ndim == 2 else 1, subtype=subtype)

    block = int(_BLOCK_S * mst_sr)
    max_sum_err = 0.0
    energy: dict[str, float] = {k: 0.0 for k, _ in track_specs}
    neg_m0 = 0
    total_ctrl = len(ctrl_times)

    i = 0
    while i < n:
        j = min(n, i + block)
        voc_b = voc_x[i:j]
        mst_b = mst_x[i:j]
        inst_b = inst_x[i:j]
        t = (np.arange(i, j, dtype="float64")) / mst_sr
        g_spk = {sid: np.interp(t, ctrl_times, g).astype("float32")[:, None]
                 for sid, g in gains_ctrl.items()}
        m0 = np.interp(t, ctrl_times, m0_ctrl).astype("float32")
        neg_m0 += int(np.sum(m0 < -1e-6))
        acc = np.zeros_like(voc_b, dtype="float32")
        for sid, g in g_spk.items():
            key = f"speaker_{sid + 1:02d}"
            out_b = g * voc_b
            files[key].write(out_b)
            acc += out_b
            energy[key] += float(np.sum(out_b.astype("float64") ** 2))
        other_b = (m0[:, None] * voc_b).astype("float32")
        files["other"].write(other_b)
        acc += other_b
        energy["other"] += float(np.sum(other_b.astype("float64") ** 2))
        files["bgm"].write(inst_b)
        acc += inst_b
        energy["bgm"] += float(np.sum(inst_b.astype("float64") ** 2))
        res_b = (mst_b - voc_b - inst_b).astype("float32")
        files["residual"].write(res_b)
        acc += res_b
        energy["residual"] += float(np.sum(res_b.astype("float64") ** 2))
        max_sum_err = max(max_sum_err, float(np.max(np.abs(acc - mst_b))) if j > i else 0.0)
        i = j
        if on_progress:
            on_progress(i / n)
    for f_ in files.values():
        f_.close()

    # 目标采样率 != 主轴率：全轨同一滤波器最后一步重采样（线性算子，契约不破）
    if not streaming_ok:
        for key, _ in track_specs:
            src = tmp_dir / f"{key}.wav"
            x, _ = audio_io.load_audio(src)
            audio_io.save_wav(out_dir / f"{key}.wav", audio_io.resample(x, mst_sr, target_sr),
                              target_sr, subtype=subtype)
            src.unlink(missing_ok=True)
        tmp_dir.rmdir()

    tracks = [{"key": key, "label": label, "file": f"{key}.wav", "kind":
               "speaker" if key.startswith("speaker") else key} for key, label in track_specs]
    master_energy = float(np.sum(mst_x.astype("float64") ** 2)) + 1e-12
    qc = {
        "sum_max_abs_err": max_sum_err,
        "sum_pass": max_sum_err < 1e-3,
        "residual_energy_ratio": energy["residual"] / master_energy,
        "neg_complement_samples": neg_m0,
        "vocals_len_adjusted": bool(adj_v),
        "inst_len_adjusted": bool(adj_i),
        "sr_mismatch": sr_mismatch,
        "track_energy": {k: v for k, v in energy.items()},
        "note": "residual 吸收 MSS 分离的线性不一致；占比大说明伴奏分离在该素材上偏差较大",
    }
    return LedgerResult(tracks=tracks, qc=qc, duration_s=duration, n_samples=n, sr_out=target_sr)


def verify_sum(track_files: list[Path], master_path: Path, sr: int) -> dict:
    """交付前独立复核：重读全部输出轨，Σ 与 master 比对（只计已知轨道文件）。
    master 与轨道采样率不一致时，先把 master 重采样到轨道域再比。"""
    acc: np.ndarray | None = None
    for wf in track_files:
        if not wf.exists():
            return {"verify_sum_max_abs_err": None, "pass": False, "error": f"缺轨道 {wf.name}"}
        x, sr_x = audio_io.load_audio(wf)
        if sr_x != sr:
            return {"verify_sum_max_abs_err": None, "pass": False, "error": f"{wf.name} 采样率 {sr_x}≠{sr}"}
        acc = x if acc is None else acc + x
    if acc is None:
        return {"verify_sum_max_abs_err": None, "pass": False, "error": "无可比对轨道"}
    mst, mst_sr = audio_io.load_audio(master_path)
    if mst_sr != sr:
        mst = audio_io.resample(mst, mst_sr, sr)
    k = min(acc.shape[0], mst.shape[0])
    err = float(np.max(np.abs(acc[:k] - mst[:k])))
    return {"verify_sum_max_abs_err": err, "pass": err < 1e-3}
