# PHOTON × Mamba2 — プロジェクト総括（区切り時点: 2026-07）

Rust 製の実験的 LLM エンジン。**PHOTON**（Fujitsu × 理研 AIP）の階層型自己回帰
アーキテクチャを骨格に、各レベルの local autoregressive 部を **Mamba2 SSD**
（State Space Dual）に置換したハイブリッドを、**バックエンド非依存のコア + 差し替え
可能な計算バックエンド**という設計で実装し、その価値を統制実験で検証したもの。

本書はプロジェクトを一区切りとした時点での**完結した現状総括**。個別の実測ログや
論文式番号との対応は開発時の内部ドキュメントに残しているが、結論と根拠はここに
自己完結でまとめてある。

---

## 0. 一行でいうと

**メモリ効率という工学目標は達成した（100M モデルを 12GB の民生 GPU で学習可能、
自前 CUDA バックエンドも動作）。しかし「PHOTON 階層が flat な Mamba2 スタックより
良い言語モデルになる」という仮説は、小規模では成り立ったが、スケールと多様な
コーパスでは逆転した（下記 §3）。** これは正直な中間結論であり、PHOTON×Mamba2 の
組み合わせ、あるいはこの学習アルゴリズムとの相性に、スケールで表面化する問題が
あることを示している。

---

## 1. アーキテクチャ

### 1.1 PHOTON 骨格 + Mamba2 SSD

- **PHOTON**: 階層型（level 0..L）の自己回帰モデル。上位 level が短い系列上で
  「圧縮された」表現を生成し、下位 level がそれを条件に局所的な系列を展開する。
  hierarchical encoder ×2 + decoder ×1 の 3 スタック構成。
- **置換**: PHOTON の各 level の local autoregressive モジュールを Mamba2 の
  **SSD（対角 A、§6）** ブロックに置換。selective scan (Mamba1) は採用せず、
  SSD（対角）のみに統一（数値安定性と実装の一貫性のため）。
- **狙い**: 階層化による KV/state トラフィック削減（PHOTON の主張）× Mamba2 の
  線形時間長系列処理を両取りする。

### 1.2 バックエンド抽象化（最重要の設計不変条件）

- **`pm-core` は Candle / cudarc / Tenstorrent SDK のいずれにも依存しない。**
  モデル定義は `Tensor` / `Ops` / `Backend` trait 越しで完結。
- バックエンド追加は **trait 実装のみ**で完結し、`pm-train` / `pm-infer` を
  書き換えない設計。
- **数値同等性**を規約化: 同じ重み・入力に対し fp32 で 1e-4 以内、fp16 で 1e-2
  以内。逸脱はバグ扱い。
- object safety より**単相化（ジェネリクス）**を優先（SSD scan 等ホットパスの
  vtable コスト回避）。

### 1.3 クレート構成

| crate | 役割 |
| --- | --- |
| `pm-core` | モデル定義・trait（`Backend`/`Ops`/`Tensor`/`Dtype`）。バックエンド非依存 |
| `pm-backend` | バックエンド共通の補助 |
| `pm-candle` | Candle ベースの**参照実装**（数値の ground truth） |
| `pm-cuda` | 自前 CUDA バックエンド（cudarc + 手書き PTX カーネル、autograd tape） |
| `pm-data` | コーパス供給（テキスト → トークン packing、ストリーミング対応） |
| `pm-tokenizer` | GPT-2 BPE トークナイザ |
| `pm-train` | 学習ループ（AdamW、勾配クリップ、activation checkpointing） |
| `pm-infer` | 推論 |
| `pm-cli` | `pm train / generate / eval / bench-*` の CLI |

---

## 2. 実装できたもの（工学的成果）

- **自前 CUDA バックエンド `pm-cuda`**: cudarc 経由で手書き PTX カーネル
  （SSD scan、depthwise conv1d、fused cross-entropy、fused PHOTON loss、
  broadcast matmul 等）を駆動し、独自の reverse-mode autograd tape を持つ。
  Candle 参照実装と数値一致を保ちつつ、Candle のメモリ律速・ホスト往復を回避。
- **メモリ効率**: hierarchical-level 単位の **activation checkpointing** により、
  102M パラメータ・T=2048 の学習が 12GB VRAM 内に収まる（north star 達成）。
- **速度最適化（Phase B'）**: ホスト往復の除去、broadcast op の device 実装、
  narrow/mul/step の融合などで学習 step を **約 6×** 改善。
- **ストリーミング学習ハーネス**: FineWeb-Edu を HuggingFace から FIFO 経由で
  垂れ流し、既存のファイル読み取り経路にそのまま流し込む（コーパスをディスクに
  永続化しない・no-shuffle で決定論的 = 2 run 間の token-match を保証）。
- **評価**: HellaSwag（全 10,042 問）、held-out cross-entropy。

---

## 3. 実験結果（統制実験）

**同一パラメータ数（±0.74%）・同一データ・同一 step 数・同一 seed / lr / clip、
差分はアーキテクチャのみ**、という統制のもとで PHOTON α=0 と flat Mamba2（32 層）を
比較した。両者とも pure cross-entropy（PHOTON は α=0 なので補助 loss L_rec は
係数 0 = CE のみ）で、**公平な CE 対 CE 比較**。

| 実験 | データ / 規模 | PHOTON 最終 CE | flat 最終 CE | Δ = flat − PHOTON |
| --- | --- | ---: | ---: | ---: |
| **D.2b** | TinyStories / 5.5M tok | **1.698** | 2.554 | **+0.856 nat（PHOTON 勝ち）** |
| **D.4** | FineWeb-Edu / 60M tok | 7.088 | **5.797** | **−1.291 nat（flat 勝ち）** |

**= スケール（10.8×）と多様ドメインへの拡大で順位が逆転した。**

### 3.1 何が起きたか

- D.4 では PHOTON の loss が **step ~800 で ~7.4 に停滞**し、残り 28,500 step で
  ほぼ改善しなかった。一方 flat は滑らかに ~5.8 まで降下し続けた。実行の 25%
  時点では両者ほぼ同点で、そこから flat だけが伸びた。
- PHOTON の **grad_norm は終始 ~147（flat の約 60×、clip=1.0 に毎 step ハード
  クリップ）** という**最適化不安定**を示した。flat は grad_norm ~2.4 で安定。
- **HellaSwag は両者チャンス（25.2% / 24.8%）** で、60M tok 規模では信号なし
  （合否ゲートにはしない）。

### 3.2 解釈（正直な中間結論）

この逆転は **scale と domain が同時に変わった交絡**であり、どちらが主因かは
本実験だけでは切り分けられない。ただし grad_norm の恒常的な暴れは、少なくとも
**PHOTON がこの lr（6e-4）/ 規模で under-tuned**（学習アルゴリズムとの相性が悪い）
であることを示す。より踏み込むと、**PHOTON 階層と Mamba2 SSD の組み合わせ自体、
あるいは階層構造と勾配ベース学習の相性**に、小規模・構造化ドメインでは表面化
しないが、大規模・多様テキストで顕在化する問題がある、というのが現時点の見立て。

**本実験は「PHOTON が本質的に劣る」ことの証明ではない**（lr sweep / warmup で
回復する可能性は残る）。しかし「このアーキ + この学習設定では、スケールで flat に
負け、かつ最適化が不安定になる」ことは確かに観測された。

---

## 4. メモリ north-star（達成）

| | flat Mamba2 (no-ckpt) | PHOTON α=0 (ckpt) |
| --- | ---: | ---: |
| peak VRAM | ~7.2 GB | ~8.4 GB |
| throughput | 550 tok/s | 540 tok/s |

**両モデルとも 12GB カード内で 102M を 60M tok 学習完了。** PHOTON は
hierarchical-level checkpointing 必須（no-ckpt では B=4 で OOM）だが、ckpt の
再計算コストは PHOTON の軽い per-token 計算にほぼ隠れ、**flat（no-ckpt）と同等の
throughput** を出す。**メモリ効率という工学目標は達成**——ただし §3 の通り、
メモリ効率それ自体は品質優位ではない。

---

## 5. 途中で潰した厄介なバグ（工学的教訓）

1. **Candle の no-backward op**: `candle_nn::ops` の `rms_norm` 等は backward を
   持たず、学習中 embedding + out_proj **しか更新されていなかった**。
   differentiable な実装に差し替えて解決。→ 新 op は forward だけでなく
   **全 param に勾配が届くか**を必ず検査する規約に。
2. **nvptx カーネルの実機ハング**: 縮約ループ境界をコンパイル時定数にすると
   LLVM が完全アンロールし、`bar.sync` が divergent な 2 箇所に分裂して**実機で
   8 時間ハング**。境界を実行時値にしてアンロール不能にすることで解決。
3. **activation checkpointing の勾配破壊**: 自前 tape は backward 毎に clear する
   が、checkpoint 学習は 1 step で backward を複数回呼ぶため、キャッシュした
   `NodeId` が世代跨ぎで別エントリを誤指しし、PHOTON の 48/54 param が勾配ゼロに
   なっていた（D.4 初回の PHOTON が「負けた」真因）。`NodeId{generation, index}`
   + 遅延再登録で解決し、no-ckpt と bit 一致を検証。

---

## 6. フェーズ状況

| Phase | スタック | 状況 |
| --- | --- | --- |
| 1 | Rust + Candle / RTX 5070 | 参照実装・学習パイプライン完成 |
| 1.5 (B') | Rust + cudarc + 自前 PTX | **自前 CUDA バックエンド動作・6× 高速化・本比較実験の主戦場** |
| 2 | 自前 PTX の本格融合カーネル | 一部着手（`nvptx_example/`） |
| 3 | Tenstorrent p100a | 未着手（ハードウェア未到着） |

---

## 7. 限界と未解決の問い

- **逆転は scale × domain の交絡**。切り分けには PHOTON を TinyStories で 60M tok、
  または FineWeb-Edu で 5.5M tok、のいずれか一変数だけ動かした実験が要る。
- **PHOTON は under-tuned の可能性**。lr sweep（3e-4 / 1e-4）+ warmup を試さずに
  「アーキが劣る」と断ずるのは早計。ただしそれで回復する保証もない。
- **HellaSwag は 60M tok では無力**（~10B tok 級が必要）。品質の主指標は
  loss-gap であって HellaSwag ではない。
- **1B tok は未実施**（~540 tok/s では ~21 日、単一 GPU で非現実的）。Phase 2 の
  高速化以降に持ち越し。
- **単一 seed**。ただし 1.29 nat 差は単一 seed のノイズを大きく超える。
- **参照コーパスは frozen artifact ではなくストリーミング**。byte 単位の再現は
  HF の shard 構成と no-shuffle 逐次消費に依存。

---

## 8. ビルド / 実行

```bash
# ビルド（自前 CUDA バックエンドは --features cuda 必須、nvcc を PATH に）
PATH=/opt/cuda/bin:$PATH CUDA_HOME=/opt/cuda \
  cargo build -p pm-cli --features cuda --release

# pm-core がバックエンドに依存していないことを検査（CI でも実行）
cargo tree -p pm-core --edges normal | grep -E '(candle|cudarc)' && exit 1 || echo OK

# 学習 / 生成 / 評価
./target/release/pm train    --backend cuda --config configs/<...>.toml
./target/release/pm generate --backend cuda --model checkpoints/<...>.safetensors --prompt "..."
./target/release/pm eval hellaswag --backend cuda --config <...> --model <...> --data <...>
```

環境: Linux (CachyOS) / NVIDIA RTX 5070 12GB, sm_120 (Blackwell) / CUDA 13.3 /
Rust stable（Phase 2 で nvptx64 用に nightly 併用）。

---

## 9. ライセンスと出典

- 本リポジトリのコードは **BSD 3-Clause License**（`LICENSE`）。© 2026 Tom039224.
- 参照論文（本リポジトリには**含めない**）:
  - PHOTON — *Hierarchical Autoregressive Modeling for Lightspeed and
    Memory-Efficient Language Generation*（Fujitsu, RIKEN AIP）
  - Mamba-2 — Dao & Gu, *Transformers are SSMs: Generalized Models and
    Efficient Algorithms Through Structured State Space Duality*
- 評価・比較に用いた TinyStories / FineWeb-Edu / GPT-2 tokenizer / HellaSwag は
  それぞれの配布元のライセンスに従う第三者データであり、本リポジトリには含めない。
