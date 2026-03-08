## Dual-Channel Graph Link Prediction (Low-pass / High-pass + Adaptive MoE)

该项目实现了用于图结构数据链路预测的双通道编码器：
- **低通道（low-pass）**：捕捉同配（homophily）平滑信号；
- **高通道（high-pass）**：捕捉异配（heterophily）差异信号；
- **自适应 MoE 解码器**：针对每条候选边动态选择 low/high/mix 专家并加权融合。

支持数据集：
- Heterophilic: `texas`, `cornell`, `wisconsin`, `chameleon`, `squirrel`
- Homophilic: `cora`, `citeseer`, `pubmed`

## 运行

```bash
pip install -r requirements.txt
python train_dual_channel_lp.py --decoder moe
```

对比普通解码器：

```bash
python train_dual_channel_lp.py --decoder dot
python train_dual_channel_lp.py --decoder mlp
```

## 日志输出

训练完成后会自动保存本地日志到 `logs/`：
- `dual_channel_lp_*.jsonl`：逐次运行结果（按 seed 与 dataset）
- `dual_channel_lp_*_summary.csv`：按数据集聚合的均值与标准差

可用 `--log-dir` 修改日志目录。
