# Photon × Mamba2 LLM（日本語）

English: **[README.md](./README.md)**

Rust 製の実験的 LLM エンジン。**PHOTON**（Fujitsu × 理研 AIP）の階層型自己回帰
アーキテクチャを骨格に、各レベルの local autoregressive モジュールを **Mamba2** の
SSD (State Space Dual) ブロックに置換したハイブリッドを、**バックエンド非依存の
コア + 差し替え可能な計算バックエンド**という設計で実装・検証したもの。

> **状況（2026-07, 一区切り）.** 工学目標であるメモリ効率は達成（102M モデルを
> 12GB の民生 GPU で学習可能・自前 CUDA バックエンド動作）。ただし研究仮説
> ——「PHOTON 階層が flat な Mamba2 スタックより良い言語モデルになる」——は、
> 小規模では成り立ったが**スケールで逆転**した。根拠を含む詳細は
> **[SUMMARY.md](./SUMMARY.md)** に自己完結でまとめてある。

> 🤖 **AI で開発.** 本プロジェクトは AI コーディングツール（Anthropic の Claude /
> Claude Code）を全面的に活用して開発した。アーキテクチャ、カーネル・autograd
> 実装、実験設計、そして下記の正直な分析は、人間 + AI のループで作られている。

## 主要結果

同一パラメータ数（±0.74%）・同一データ・同一 step 数・同一 seed / lr / clip、
**差分はアーキテクチャのみ**の統制実験。両者とも pure cross-entropy で比較
（PHOTON は α=0 = 補助 loss 係数 0 なので CE のみ）。

| 実験 | データ / 規模 | PHOTON CE | flat CE | Δ = flat − PHOTON |
| --- | --- | ---: | ---: | ---: |
| **D.2b** | TinyStories / 5.5M tok | **1.698** | 2.554 | **+0.86 nat（PHOTON 勝ち）** |
| **D.4** | FineWeb-Edu / 60M tok | 7.088 | **5.797** | **−1.29 nat（flat 勝ち）** |

スケール（約 10.8×）と多様コーパスへの拡大で**順位が逆転**。D.4 では PHOTON の
loss が step ~800 で停滞し、flat だけが降下を続けた。PHOTON の grad_norm は約 60×
（mean 147 vs 2.37）で毎 step ハードクリップ = 最適化不安定。これは「PHOTON が
本質的に劣る」証明ではない（lr sweep / warmup は未試行）が、小規模の優位が
**転移せず**、このアーキ + 学習設定がスケールで不安定になることは確かに観測された。
交絡と未解決の問いは `SUMMARY.md` を参照。

## 設計の柱

- **Backend 抽象化を最優先**: コアのモデル定義（`pm-core`）は Candle / cudarc /
  Tenstorrent SDK のいずれにも依存しない。すべて `Tensor` / `Ops` / `Backend`
  trait 越し。バックエンド追加は trait 実装のみ・数値同等性（fp32 1e-4 以内）を規約化。
- **段階的最適化**: Phase 1 Candle 参照実装 → Phase 1.5/2 自前 CUDA バックエンド
  （cudarc + 手書き PTX + autograd tape）→ Phase 3 Tenstorrent へ再写像。
- **学習込み**: activation checkpointing を前提に最初から設計。

## クレート構成

`pm-core`（trait / モデル定義・バックエンド非依存）・`pm-candle`（参照実装）・
`pm-cuda`（自前 CUDA バックエンド）・`pm-data` / `pm-tokenizer` / `pm-train` /
`pm-infer` / `pm-cli`。

## ビルド / 実行

```bash
# 自前 CUDA バックエンドは --features cuda 必須（nvcc を PATH に）
PATH=/opt/cuda/bin:$PATH CUDA_HOME=/opt/cuda \
  cargo build -p pm-cli --features cuda --release

# pm-core がバックエンドに依存していないことを検査（CI でも実行）
cargo tree -p pm-core --edges normal | grep -E '(candle|cudarc)' && exit 1 || echo OK

./target/release/pm train --backend cuda --config configs/<...>.toml
```

環境: Linux (CachyOS) / NVIDIA RTX 5070 12GB, sm_120 (Blackwell) / CUDA 13.3 /
Rust stable（Phase 2 で nvptx64 用に nightly 併用）。

## ライセンス

**BSD 3-Clause License**（[LICENSE](./LICENSE)）。© 2026 Tom039224.

参照論文（PHOTON, Mamba-2）および評価データ（TinyStories / FineWeb-Edu /
GPT-2 tokenizer / HellaSwag）は第三者の著作物であり、本リポジトリには含めない。
出典は [SUMMARY.md](./SUMMARY.md) §9 を参照。
