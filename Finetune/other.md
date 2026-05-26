SFT（监督微调）

指令微调（Instruction tuning）

多任务微调

参数高效微调（PEFT）

LoRA / QLoRA / Adapter / Prefix tuning

Fine-tuning（微调）
│
├── 1. 训练目标范式（Objective）
│     ├── SFT（监督微调）
│     │     └── Instruction Tuning（指令微调）
│     └── Continued Pretraining（继续预训练 / 领域自适应）
│
├── 2. 参数更新方式（Parameterization）
│     ├── Full Fine-tuning（全参数）
│     └── PEFT
│           ├── LoRA
│           ├── Adapter
│           ├── Prompt Tuning
│           ├── Prefix Tuning
│           └── P-Tuning / P-Tuning v2
│
└── 3. 训练策略扩展（Training Strategy）
      ├── Multi-task Fine-tuning
      ├── Curriculum Learning
      ├── Instruction Mixing
      └── Distillation-based Fine-tuning