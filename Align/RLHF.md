# 基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF)

基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF) 是最经典的对齐流程. 一般分为三步:

## 1. SFT

这一步即是 Finetune 中进行的步骤, 旨在利用高质量指令数据训练 PLM, **提供一个可用的初始策略**. 只需要让模型学会基本的回答格式和指令遵循.

## 2. Reward Model: 训练奖励模型

人类不直接给每个回答一个绝对分数，而是更常见地比较两个回答：

由于人类相比于直接将偏好量化成奖励分数, 更擅长根据偏好对回答进行排序, 故构造奖励模型数据时采用成对比较的方式, 给定 prompt $x$, 模型生成两个回答 $y_1, y_2$, 让人类标注哪个更好.

用这些排序数据训练一个奖励模型 (RM): $r_\phi(x,y)$, 使得:

$$
r_\phi(x,y_w) > r_\phi(x,y_l)
$$
输出一个标量分数, 通过优化以下损失函数实现:

$$
\mathcal{L}_{RM}
= -\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
$$
让 RM 学习 "人类偏好排序".

## 3. 强化学习优化 (Reinforcement Learning, RL)

使用强化学习算法 (通常是 **[PPO](./PPO.md)** 等), 根据 RM 的打分来优化模型. 基本步骤如下:

1. 给模型 (Actor) 输入一个 Prompt，模型生成一个 Response.
2. 将这个 Response 送给奖励模型 (RM), RM 给出一个分数 (Reward).
3. 模型根据这个分数, 利用强化学习算法更新自身参数, 鼓励以后多生成高分回答.
