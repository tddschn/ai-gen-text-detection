License: CC BY 4.0
arXiv:2306.08543v2 [cs.CL] 28 Feb 2024
MiniLLM: Knowledge Distillation of Large Language Models
Yuxian Gu
1
,
2
,    Li Dong
2
,    Furu Wei
2
,    Minlie Huang
1

1
The CoAI Group, Tsinghua University
2
Microsoft Research
guyx21@mails.tsinghua.edu.cn    {lidong1,fuwei}@microsoft.com
aihuang@tsinghua.edu.cn
Contribution during an internship at Microsoft Research.Corresponding author.
Abstract
Knowledge Distillation (KD) is a promising technique for reducing the high computational demand of large language models (LLMs). However, previous KD methods are primarily applied to white-box classification models or training small models to imitate black-box model APIs like ChatGPT. How to effectively distill the knowledge of white-box LLMs into small models is still under-explored, which becomes more important with the prosperity of open-source LLMs. In this work, we propose a KD approach that distills LLMs into smaller language models. We first replace the forward Kullback-Leibler divergence (KLD) objective in the standard KD approaches with reverse KLD, which is more suitable for KD on generative language models, to prevent the student model from overestimating the low-probability regions of the teacher distribution. Then, we derive an effective optimization approach to learn this objective. The student models are named MiniLLM. Extensive experiments in the instruction-following setting show that MiniLLM generates more precise responses with higher overall quality, lower exposure bias, better calibration, and higher long-text generation performance than the baselines. Our method is scalable for different model families with 120M to 13B parameters. Our code, data, and model checkpoints can be found in https://github.com/microsoft/LMOps/tree/main/minillm.

Refer to caption
Figure 1:The comparison of MiniLLM with the sequence-level KD (SeqKD; 36, 67, 15, 53, 22, 81) in terms of the average GPT-4 feedback score on our evaluation sets. Left: GPT-2-1.5B as the teacher model and GPT-2 125M, 340M, 760M as the student models. Middle: GPT-J 6B as the teacher model and GPT-2 760M, 1.5B, GPT-Neo 2.7B as the student models. Right: OPT 13B as the teacher and OPT 1.3B, 2.7B, 6.7B as the student models.
1Introduction
With the rapid development of large language models (LLMs; 7, 29, 4, 16, 50), a common technique to reduce their high computational resource demand is knowledge distillation (KD; 28), where we train a small student model with supervision from a large teacher model. Two categories of KD are commonly applied: black-box KD, where only the teacher-generated texts are accessible, and white-box KD, where the teacher model’s output distribution or intermediate hidden states are also available [30]. Recently, black-box KD has shown promising results in fine-tuning small models on the prompt-response pairs generated by LLM APIs [67, 15, 79, 53]. With the emergence of more open-source LLMs [83, 68], white-box KD becomes more valuable for both research communities and industry sectors because student models receive better signals from the output distribution and hidden states of teacher models, thereby potentially resulting in higher performance. However, white-box KD approaches are mostly studied for small (
<
 1B parameters) language understanding models [58, 78], while white-box KD for LLMs is yet to be explored.

In this work, we investigate white-box KD of LLMs where the output distribution of the teacher model is available. We argue that the standard KD objectives [36, 63, 15, 67] are sub-optimal for LLMs that perform tasks in a generative manner. Given the teacher distribution 
𝑝
⁢
(
𝒚
|
𝒙
)
 and the student distribution 
𝑞
𝜃
⁢
(
𝒚
|
𝒙
)
 parameterized by 
𝜃
, standard KD objectives (including several variants for sequence-level models) essentially minimize the approximated forward Kullback-Leibler divergence (KLD) between the teacher and the student distribution, termed as 
KL
[
𝑝
|
|
𝑞
𝜃
]
, which forces 
𝑞
𝜃
 to cover all modes of 
𝑝
. For text classification tasks, 
KL
[
𝑝
|
|
𝑞
𝜃
]
 works well because the output space usually consists of a finite number of classes such that both 
𝑝
⁢
(
𝒚
|
𝒙
)
 and 
𝑞
𝜃
⁢
(
𝒚
|
𝒙
)
 have few modes. However, for open-ended text generation tasks, which is usually the case of LLM applications, the output spaces are much more complex and 
𝑝
⁢
(
𝒚
|
𝒙
)
 can contain many more modes than what 
𝑞
𝜃
⁢
(
𝒚
|
𝒙
)
 can express due to the limited model capacity. Minimizing forward KLD causes 
𝑞
𝜃
 to assign unreasonably high probabilities to the void regions of 
𝑝
 [46] and produces very unlikely samples under 
𝑝
 during free-run generation [27].

Refer to caption
Figure 2:The toy experiment. We fit a Gaussian mixture distribution with a single Gaussian distribution using forward KLD and reverse KLD.
To alleviate this problem, we propose to minimize reverse KLD, 
KL
[
𝑞
𝜃
|
|
𝑝
]
, widely used in computer vision [43] and reinforcement learning [17]. Compared to 
KL
[
𝑝
|
|
𝑞
𝜃
]
, minimizing 
KL
[
𝑞
𝜃
|
|
𝑝
]
 causes 
𝑞
𝜃
 to seek the major modes of 
𝑝
, and assign low probabilities to 
𝑝
’s void regions [44], as illustrated in Table 2 and discussed in Section 2.1. In LLM text generation, this means that the student model avoids learning too many long-tail variants [23] in the teacher model’s distribution and focuses on the correctness of the generated cotents, which is critical in practical scenarios that require truthfulness and reliability [33]. To optimize 
min
𝜃
KL
[
𝑞
𝜃
|
|
𝑝
]
, as shown in Section 2.2, we derive the gradient of the objective with Policy Gradient [60]. To further stabilize and accelerate training, we propose (1) single-step decomposition to reduce variance, (2) teacher-mixed sampling to alleviate reward hacking, and (3) length normalization to eliminate the length bias. Finally, we introduce the overall KD algorithm in Section 2.3. Our student models are named MiniLLM, indicating our method is suitable and works well for compressing large (generative) language models.

We apply our method to various generative language models [56, 83, 68] with sizes ranging from 120M to 13B in the instruction-following setting [65, 71] that covers a large range of NLP tasks. We use 5 datasets with Rouge-L [39], the GPT-4 feedback, and human judgment for evaluation. Experiments show that MiniLLM consistently outperforms standard KD baselines on all the datasets and scales up well from 120M to 13B models (see Figure 1). More analysis shows that MiniLLM yields lower exposure bias, better calibration, and higher long response generation performance, with neglectable loss of diversity.

2Method
We consider conditional text generation where the model produces a response 
𝒚
=
{
𝑦
𝑡
}
𝑡
=
1
𝑇
 conditioning on a prompt 
𝒙
 sampled from the distribution 
𝑝
𝒙
, which is typically how LLMs perform tasks. We formulate KD as an optimization problem to minimize the difference between a fixed teacher model distribution 
𝑝
⁢
(
𝒚
|
𝒙
)
 and a student model distribution 
𝑞
𝜃
⁢
(
𝒚
|
𝒙
)
 parameterized by 
𝜃
. The standard KD methods approximately1 minimize the forward KLD: 
KL
[
𝑝
|
|
𝑞
𝜃
]
=
𝔼
𝒙
∼
𝑝
𝒙
,
𝒚
∼
𝑝
′
log
𝑝
⁢
(
𝒚
|
𝒙
)
𝑞
𝜃
⁢
(
𝒚
|
𝒙
)
, where 
𝑝
′
 can be real data distribution (word-level KD) or teacher distribution 
𝑝
 (sequence-level KD). Though widely used, 
KL
[
𝑝
|
|
𝑞
𝜃
]
 tends to overestimate the void regions of 
𝑝
 in text generation tasks when 
𝑞
𝜃
 is insufficiently expressive [32]. KD for LLMs fits the case because LLMs perform tasks in a generative manner, such that the low-capacity student models cannot perfectly imitate the complex text generation distribution of the teacher models or humans.

2.1MiniLLM: Knowledge Distillation with Reverse KLD
We consider minimizing the reverse KLD between the student and teacher model distributions as the learning objective for MiniLLM:

𝜃
=
arg
⁡
min
𝜃
⁡
ℒ
⁢
(
𝜃
)
=
arg
min
𝜃
KL
[
𝑞
𝜃
|
|
𝑝
]
(1)
=
arg
⁡
min
𝜃
⁡
[
−
𝔼
𝒙
∼
𝑝
𝒙
,
𝒚
∼
𝑞
𝜃
log
⁡
𝑝
⁢
(
𝒚
|
𝒙
)
𝑞
𝜃
⁢
(
𝒚
|
𝒙
)
]
.
Minimizing reverse KLD has been shown to cause the mode-seeking behavior in generative modeling [27, 47, 11, 43], where 
𝑞
𝜃
 assigns high probabilities to 
𝑝
’s large modes and ignore the small ones (illustrated in a toy experiment in Figure 2). In this work, we first study this property for KD of LLMs in text generation. Minimizing forward KLD causes 
𝑞
𝜃
 to place large probability masses on the zero-probability regions of 
𝑝
, corresponding to the generation of low-quality text in practice, while reverse KLD focuses on 
𝑝
’s major modes, which is crucial to ensure the correctness and faithfulness of text generation. As illustrated in Figure 3, unlike sequence-level KD that minimizes forward KLD [36, 67], MiniLLM that minimizes reverse KLD does not force 
𝑞
𝜃
 to fit all 
𝒚
 sampled from the teacher distribution 
𝑝
. Instead, it encourages the student to generate samples preferred by the teacher within its own capacities, which is more possible to achieve. Interestingly, we also find another perspective of understanding MiniLLM motivated by Inverse Reinforcement Learning [82]. We present the related derivation in Appendix A.1.

Refer to caption
Figure 3:Comparison between sequence-level KD (left) and MiniLLM (right). Sequence-level KD forces the student to memorize all samples generated by the teacher model, while MiniLLM improves its generated texts with the teacher model’s feedback.
2.2Optimization with Policy Gradient
Gradient Derivation
We notice that the gradient of the objective function 
ℒ
⁢
(
𝜃
)
 in Equation (1) can be derived using the Policy Gradient Theorem [72, 26]:

∇
ℒ
⁢
(
𝜃
)
=
−
𝔼
𝒙
∼
𝑝
𝒙
,
𝒚
∼
𝑞
𝜃
(
⋅
|
𝒙
)
∑
𝑡
=
1
𝑇
(
𝑅
𝑡
−
1
)
⁢
∇
log
⁡
𝑞
𝜃
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
,
(2)
where 
𝑇
=
|
𝒚
|
 and 
𝑅
𝑡
=
∑
𝑡
′
=
𝑡
𝑇
log
⁡
𝑝
⁢
(
𝑦
𝑡
′
|
𝒚
<
𝑡
′
,
𝒙
)
𝑞
𝜃
⁢
(
𝑦
𝑡
′
|
𝒚
<
𝑡
′
,
𝒙
)
 is the accumulation of 
𝑟
𝑡
′
=
log
⁡
𝑝
⁢
(
𝑦
𝑡
′
|
𝒚
<
𝑡
′
,
𝒙
)
𝑞
𝜃
⁢
(
𝑦
𝑡
′
|
𝒚
<
𝑡
′
,
𝒙
)
 that measures the quality of each step’s generation. Intuitively, the generated texts are supposed to have high probabilities under the teacher distribution by increasing 
𝑝
⁢
(
𝑦
𝑡
′
|
𝒚
<
𝑡
′
,
𝒙
)
, but simultaneously stay diverse by lowering 
𝑞
𝜃
⁢
(
𝑦
𝑡
′
|
𝒚
<
𝑡
′
,
𝒙
)
. The expectation in Eq. 2 is computed by Monte-Carlo sampling. Full derivation can be found in Appendix A.2. However, policy gradient suffers from high variance and reward hacking [59], despite some subsequent solutions [64]. Besides, we notice that 
𝑅
𝑡
 favors short sentences, which causes the student model to output empty responses. Therefore, we propose three strategies to mitigate these problems.

Single-Step Decomposition
[17] has found that the single-step generation quality 
𝑟
𝑡
 is critical to the training variance because the error in the front tokens accumulates along the whole sentence. To pay more attention to 
𝑟
𝑡
, we re-write 
∇
ℒ
⁢
(
𝜃
)
 to decompose 
𝑟
𝑡
 from 
𝑅
𝑡
 and directly compute the gradient of 
𝔼
𝑦
𝑡
∼
𝑞
𝜃
⁢
(
𝑡
)
[
𝑟
𝑡
]
 (see Appendix A.3 for the full derivation):

∇
ℒ
⁢
(
𝜃
)
=
𝔼
𝒙
∼
𝑝
𝒙
𝒚
∼
𝑞
𝜃
(
⋅
|
𝒙
)
[
−
∑
𝑡
=
1
𝑇
∇
⁢
𝔼
𝑦
𝑡
∼
𝑞
𝜃
⁢
(
𝑡
)
[
𝑟
𝑡
]
]
+
𝔼
𝒙
∼
𝑝
𝒙
𝒚
∼
𝑞
𝜃
(
⋅
|
𝒙
)
[
−
∑
𝑡
=
1
𝑇
𝑅
𝑡
+
1
⁢
∇
log
⁡
𝑞
𝜃
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
]
(3)
=
(
∇
ℒ
)
Single
+
(
∇
ℒ
)
Long
,
where 
𝑞
𝜃
(
𝑡
)
=
𝑞
𝜃
(
⋅
|
𝒚
<
𝑡
,
𝒙
)
. Note that 
𝔼
𝑦
𝑡
∼
𝑞
𝜃
⁢
(
𝑡
)
[
𝑟
𝑡
]
 can be computed directly by summing over the vocabulary instead of using Monte-Carlo sampling and is derivable with respect to 
𝜃
. This decomposition gives a more precise and efficient estimation of the single-step generation quality, which reduces the variance during training and accelerates convergence.

Teacher-Mixed Sampling
We observe reward hacking [59] when training with Eq. 2 because 
𝑞
𝜃
 sometimes produces degenerated sentences 
𝒚
 that receive high scores from the teacher (e.g., repeated phrases) during sampling, especially for small student models. To create a better sampling distribution, we mix the teacher and the student distribution at each time step:

𝑝
~
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
=
𝛼
⋅
𝑝
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
+
(
1
−
𝛼
)
⋅
𝑞
𝜃
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
,
(4)
where 
𝛼
 controls the strength of the teacher mix-in. Sampling from 
𝑝
~
 suppresses low-quality generation with the teacher’s help and alleviates reward hacking. We re-write 
(
∇
ℒ
)
Single
 and 
(
∇
ℒ
)
Long
 with importance sampling to get to an unbiased estimator of the gradient [54]:

(
∇
ℒ
)
Single
=
−
𝔼
𝒙
∼
𝑝
𝒙
,
𝒚
∼
𝑝
~
(
⋅
|
𝒙
)
[
∑
𝑡
=
1
𝑇
𝑤
𝑡
⁢
∇
⁢
𝔼
𝑦
𝑡
∼
𝑞
𝜃
⁢
(
𝑡
)
[
𝑟
𝑡
]
]
,
(5)
(
∇
ℒ
)
Long
=
−
𝔼
𝒙
∼
𝑝
𝒙
,
𝒚
∼
𝑝
~
(
⋅
|
𝒙
)
[
∑
𝑡
=
1
𝑇
𝑤
𝑡
⁢
𝑅
𝑡
+
1
⁢
∇
log
⁡
𝑞
𝜃
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
]
,
where 
𝑤
𝑡
=
∏
𝑡
′
=
1
𝑡
𝑞
𝜃
⁢
(
𝑦
𝑡
′
|
𝒚
<
𝑡
′
,
𝒙
)
𝑝
~
⁢
(
𝑦
𝑡
′
|
𝒚
<
𝑡
′
,
𝒙
)
 is the importance weight. However, 
𝑤
𝑡
 brings high variance in practice because it requires multiplying per-token importance weight over multiple time steps, and thus the variance of each step accumulates. Therefore, we approximately set 
𝑤
𝑡
≈
𝑞
𝜃
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
𝑝
~
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
 to reduce the variance of the estimator in Eq. 5 [62, 40].

Length Normalization
We found that long sequences tend to have small 
𝑅
𝑡
+
1
, which encourages the model to produce short responses. Therefore, we add length normalization to 
𝑅
𝑡
+
1
 in Eq. 3:

𝑅
𝑡
+
1
Norm
=
1
𝑇
−
𝑡
−
1
⁢
∑
𝑡
′
=
𝑡
+
1
𝑇
log
⁡
𝑝
⁢
(
𝑦
𝑡
′
|
𝒚
<
𝑡
′
,
𝒙
)
𝑞
𝜃
⁢
(
𝑦
𝑡
′
|
𝒚
<
𝑡
′
,
𝒙
)
.
(6)
In Summary
Combining the strategies listed above, we have the final optimization gradient:

∇
ℒ
⁢
(
𝜃
)
=
−
𝔼
𝒙
∼
𝑝
𝒙
𝒚
∼
𝑝
~
(
⋅
|
𝒙
)
[
∑
𝑡
=
1
𝑇
𝑤
𝑡
⁢
[
∇
⁢
∑
𝑦
′
∈
𝑉
𝑞
𝜃
⁢
(
𝑦
′
|
𝒚
<
𝑡
,
𝒙
)
⁢
log
⁡
𝑝
⁢
(
𝑦
′
|
𝒚
<
𝑡
,
𝒙
)
𝑞
𝜃
⁢
(
𝑦
′
|
𝒚
<
𝑡
,
𝒙
)
⏟
(
∇
ℒ
)
Single
⁢
 part
+
𝑅
𝑡
+
1
Norm
⁢
∇
𝑞
𝜃
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
𝑞
𝜃
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
⏟
(
∇
ℒ
)
Long
Norm
⁢
 part
]
]
,
(7)
where 
𝑉
 is the vocabulary size of the language model and 
(
∇
ℒ
)
Long
Norm
 is 
(
∇
ℒ
)
Long
 with 
𝑅
𝑡
+
1
Norm
.

2.3Training Algorithm
We start from a student model pre-trained on a large long-document corpus 
𝒟
PT
. Algorithm 1 trains MiniLLM by adapting the student model to a text generation task with dataset 
𝒟
 and supervision from the teacher model, such as an LLM fine-tuned on 
𝒟
 [67, 15] or that with good task-generalization [12, 50]. In the training algorithm, we first fine-tune the student model on 
𝒟
 and pick the checkpoint with the lowest loss as an initialization for the following training. Then, we compute the gradients 
(
∇
ℒ
)
Single
 and 
(
∇
ℒ
)
Long
Norm
 based on Eq. 5 and Eq. 6, with a clipping strategy [64] added to further improve stability. Same as [51], we include a language modeling loss 
ℒ
PT
=
−
𝔼
𝒅
∼
𝒟
PT
log
⁡
𝑞
𝜃
⁢
(
𝒅
)
 to preserve the model performance on canonical NLP benchmarks. The student model is finally updated using a combination of gradients 
(
∇
ℒ
)
Single
+
(
∇
ℒ
)
Long
Norm
+
∇
ℒ
PT
. The whole training pipeline is similar to Reinforcement Learning from Human Feedback (RLHF; 51).

Algorithm 1 MiniLLM: Knowledge Distillation of LLMs
Conditional generation dataset 
𝒟
 consisting of prompts and ground-truth responses
   Pre-training corpus 
𝒟
PT
 consisting of long-document plain texts
   A teacher model with output distribution 
𝑝
   An initial student model pre-trained on 
𝒟
PT
, with the output distribution 
𝑞
𝜃
0
   Learning rate 
𝜂
;  Batch size 
𝑀
;  Clipping Threshold 
𝜖
A student model with the output distribution 
𝑞
𝜃
Fine-tune the student model from 
𝜃
0
 on 
𝒟
 supervised by the ground truth responses and choose 
𝜃
 with the lowest validation loss.
repeat
    Sample a mini-batch of prompts from 
𝒟
 and collect responses from 
𝑝
~
 to get 
𝒮
=
{
(
𝒙
𝑚
,
𝒚
𝑚
)
}
𝑚
=
1
𝑀
    Sample a mini-batch 
𝒟
′
PT
=
{
𝒅
𝑚
}
𝑚
=
1
𝑀
 from 
𝒟
PT
    Compute 
(
∇
ℒ
)
Single
=
−
1
𝑀
⁢
∑
𝒙
,
𝒚
∈
𝒮
∑
𝑡
=
1
𝑇
𝑤
𝑡
⁢
∇
⁢
∑
𝑦
𝑡
∈
𝑉
𝑞
𝜃
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
⁢
log
⁡
𝑝
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
𝑞
𝜃
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
▷
 Eq. 5
    Compute 
(
∇
ℒ
)
Long
Norm
=
−
1
|
𝑀
|
⁢
∑
𝒙
,
𝒚
∈
𝒮
∑
𝑡
=
1
𝑇
𝑅
𝑡
+
1
Norm
⁢
∇
min
⁡
[
𝜌
𝑡
⁢
(
𝜃
)
,
clip
⁡
(
𝜌
𝑡
⁢
(
𝜃
)
,
1
−
𝜖
,
1
+
𝜖
)
]
,
    where 
𝜌
𝑡
⁢
(
𝜃
)
=
𝑞
𝜃
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
𝑝
~
⁢
(
𝑦
𝑡
|
𝒚
<
𝑡
,
𝒙
)
▷
 Eq. 5, Eq. 6
    Compute the gradient of the language modeling loss: 
∇
ℒ
PT
=
−
1
𝑀
⁢
∑
𝒅
∈
𝐷
PT
′
∇
log
⁡
𝑞
𝜃
⁢
(
𝒅
)
    Update model parameters: 
𝜃
←
𝜃
−
𝜂
⁢
[
(
∇
ℒ
)
Single
+
(
∇
ℒ
)
Long
Norm
+
∇
ℒ
PT
]
until converge and return 
𝑞
𝜃
3Experiments
3.1Experimental Setup
We take instruction-following [51] as the conditional text generation task, where models are trained to generate responses according to the instructions. We fine-tune a large model on the dataset 
𝒟
 consisting of instruction-response pairs as the teacher model. Then, we compare different KD methods on 
𝒟
 by evaluating the student model’s instruction-following performance.

Base Models
Our student models come from three model families with various sizes: GPT-2 [56] (120M, 340M, 760M), OPT [83] (1.3B, 2.7B, 6.7B), and LLaMA [68] (7B). For teacher models of each model family, we use GPT-2-1.5B, OPT-13B, and LLaMA-13B respectively. These models are fine-tuned on 
𝒟
 in advance. We also present the results using GPT-J [73] as the teacher model in Appendix C.1.

Training
We construct the training data from databricks-dolly-15K2 consisting of 15K human-written instruction-response pairs. We filter out samples that exceed the context length of the models. Then, we randomly split 0.5K and 1K samples for validation and testing, respectively, leaving about 12.5K examples for training. For 
𝒟
PT
, we use OpenWebText [20] for the GPT-2 family and the RoBERTa training corpus [42] for other models. We set the teacher-mix-in strength 
𝛼
=
0.2
 throughout the experiments in Eq. 4. We use Rouge-L [39] scores on the validation set to search for hyper-parameters because it aligns better with human preference than validation losses [76]. More details are shown in Appendix B.1.

Evaluation
We evaluate the trained models on five instruction-following datasets:

• DollyEval: the 500-sample test set we split from the databricks-dolly-15k dataset.
• SelfInst [74]: A user-oriented instruction-following set with 252 samples.
• VicunaEval [15]: The 80 challenging questions used in the Vicuna evaluation.
• S-NI: The test set of Super-NaturalInstructions [76] consisting of 9K samples ranging from 119 tasks. Following [53], we split the set into 3 subsets whose ground truth response lengths lie in 
[
0
,
5
]
, 
[
6
,
10
]
, and 
[
11
,
+
∞
]
. We use the 
[
11
,
+
∞
]
 subset in Section 1 and conduct an analysis on all subsets in Section 3.3.
• UnNI: The core set of UnnaturalInstructions [25] containing 60K samples. Similar to S-NI, we first conduct the evaluations on the 
[
11
,
+
∞
]
 subset, followed by an analysis of the performance on all subsets in Appendix C.3.
We adopt three metrics to evaluate the model-generated responses:

• R-L: The Rouge-L [39] score to measure the precision of the model generation. [76] has shown that Rouge-L is suitable for large-scale instruction-following evaluation.
• GPT4: The GPT-4 feedback [80] by asking GPT-4 to compare model-generated responses with the ground truth answers3 and raise 1-10 scores for both responses (see Appendix B.2 for the prompt we use). We report the ratio of the total score of model responses and ground truth answers. This metric is only applied to DollyEval, SelfInst, and VicunaEval.
• Human Evaluation: We conduct human evaluations on the SelfInst dataset following  [53] by asking volunteers to compare two responses produced by different models and annotate “Win”, “Tie”, or “Loss”. More human evaluation details can be found in Appendix B.3.
For all test sets, we sample the responses with the temperature = 1 and report the average scores of 5 generations for each prompt with different random seeds.

Baselines
We consider three baselines in our main experiment:

• SFT w/o KD directly fine-tunes the student model on 
𝒟
 supervised by the golden responses.
• KD [58, 63] fine-tunes the student model on 
𝒟
 using the teacher distribution as the supervision at each token step, also known as word-level KD.
• SeqKD [36, 15, 67, 53, 81] fine-tunes the student model on the data generated by the teacher model.
3.2Results
Model	#Params	Method	DollyEval	SelfInst	VicunaEval	S-NI	UnNI
GPT4	R-L	GPT4	R-L	GPT4	R-L	R-L	R-L
GPT-2	1.5B	Teacher	58.4	27.6	42.9	14.3	48.6	16.3	27.6	34.9
120M	SFT w/o KD	38.6	23.3	26.3	10.0	32.8	14.7	16.3	21.4
KD	40.3	22.8	27.8	10.8	31.9	13.4	19.7	24.8
SeqKD	41.2	22.7	26.2	10.1	31.0	14.3	16.4	21.0
MiniLLM	44.7	24.6	29.2	13.2	34.1	16.9*	25.3	30.1
340M	SFT w/o KD	51.9	25.5	39.6	13.0	42.3	16.0	25.1	32.0
KD	51.6	25.0	39.2	12.0	42.8	15.4	23.7	31.0
SeqKD	50.5	25.3	39.0	12.6	43.0	16.9*	22.9	30.2
MiniLLM	52.2	25.4	40.5	15.6	42.6	17.7*	27.4	34.5
760M	SFT w/o KD	50.7	25.4	38.3	12.4	43.1	16.1	21.5	27.1
KD	53.4	25.9	40.4	13.4	43.4	16.9*	25.3	31.7
SeqKD	52.0	25.6	38.9	14.0	42.4	15.9	26.1	32.9
MiniLLM	54.7	26.4	44.6*	15.9	45.7	18.3*	29.3*	37.7*
OPT	13B	Teacher	70.3	29.2	56.1	18.4	58.0	17.8	30.4	36.1
1.3B	SFT w/o KD	52.6	26.0	37.7	11.4	40.5	15.6	23.1	28.4
KD	52.7	25.4	36.0	12.2	40.8	14.9	21.9	27.0
SeqKD	51.0	26.1	36.6	12.7	42.6	16.6	21.4	28.2
MiniLLM	60.7	26.7	47.0	14.8	50.6	17.9*	28.6	33.4
2.7B	SFT w/o KD	55.4	27.1	38.9	13.9	44.8	16.6	24.9	32.3
KD	60.5	25.9	48.6	13.8	51.3	16.7	26.3	30.2
SeqKD	57.6	27.5	40.5	13.3	44.5	16.5	25.3	32.3
MiniLLM	63.2	27.4	52.7	17.2	55.9	19.1*	30.7*	35.1
6.7B	SFT w/o KD	67.9	27.6	56.4	16.4	57.3	17.8	30.3	28.6
KD	68.6	28.3	58.0	17.0	57.0	17.5	30.7*	26.7
SeqKD	69.6	28.5	54.0	17.0	57.6	17.9*	30.4	28.2
MiniLLM	70.8*	29.0	58.5*	17.5	60.1*	18.7*	32.5*	36.7*
LLaMA	13B	Teacher	79.0	29.7	75.5	23.4	65.1	19.4	35.8	38.5
7B	SFT w/o KD	73.0	26.3	69.2	20.8	61.6	17.5	32.4	35.8
KD	73.7	27.4	70.5	20.2	62.7	18.4	33.7	37.9
SeqKD	73.6	27.5	71.5	20.8	62.6	18.1	33.7	37.6
MiniLLM	76.4	29.0	73.1	23.2	64.1	20.7*	35.5	40.2*
Table 1:Evaluation results. GPT4 and R-L stand for the average GPT-4 feedback scores and Rouge-L scores across 5 random seeds, respectively. The best scores of each model size are boldfaced, and the scores where the student model outperforms the teacher are marked with *.
We present the R-L and GPT4 evaluation results in Table 1, from which we have three observations.

First, by comparing the overall performance of MiniLLM with the baselines, we observe that the model distilled by our KD method outperforms the baselines in almost all cases, when trained with different base models, tested on various evaluation sets, and scored by both Rouge-L and GPT-4 feedback. This verifies the good generalization and high overall performance of our KD method. We also find that MiniLLM generally works much better on datasets other than Dolly compared with the baselines, indicating its good out-of-distribution generalization.

Second, the Rouge-L scores show that MiniLLM produces the most precise responses that have high overlaps with the ground-truth responses. In some cases, especially on Vicuna, S-NI, and UnNI, student models reach even higher Rouge-L scores than the teacher models, which matches the observation in [19]. We conjecture that the standard teacher-forcing fine-tuning on 
𝒟
 brings training-inference discrepancy to the teacher model, also known as exposure bias [9]. On the contrary, MiniLLM is optimized with policy optimization methods, which samples responses from student models during training and thus alleviates exposure bias [52]. We include further analysis on exposure bias in Section 3.3.

Refer to caption
Figure 4:Human evaluation results. We use LLaMA-7B as the student and LLaMA-13B as the teacher.
Third, comparing the results across model sizes and model families, we can see that the improvement of MiniLLM is consistent when the base model sizes vary from 120M to 13B across three model families. This tendency is also illustrated in Figure 1, which demonstrates the excellent scalability and generalization of our method in the era of LLMs.

The human evaluation results on the SelfInst dataset based on the LLaMA family are shown in Figure 4. MiniLLM obtains better human preference than all the baselines, performing comparably to the teacher model.

3.3Analysis
Scaling Law of Teacher
Refer to caption
Figure 5:The scaling law of teacher based on the GPT-2 family models. We compare MiniLLM and SeqKD with GPT-2-125M as the student and GPT-2 340M, 760M, and 1.5B as teachers.
Although it is intuitive that we can distill better student models from larger teacher models, [45] has shown that increasing the teacher models’ sizes does not guarantee the improvement of student models, sometimes even harming the distillation performance. It is not clear how MiniLLM works when we scale up the teacher models’ sizes. Therefore, we compare MiniLLM and SeqKD using teacher models with different sizes and fix the size of the student model. We present the results based on the GPT-2 family in Figure 5 and that based on the OPT family in Appendix C.2. We can see that MiniLLM constantly outperforms SeqKD, and the student model performance is positively correlated with the teacher model sizes. This shows the potential of our method to compress models with massive parameters.

Exposure Bias
Language generation models trained to minimize forward KLD suffer from exposure bias [9] caused by the discrepancy between teacher-forcing training and free-run generation. When training MiniLLM, the student model sees samples generated by itself, alleviating the training-inference mismatch [52]. In Figure 6, we use the ExAccErr metric [2] defined in Appendix B.5 to measure the excess accumulated error due to exposure bias. The experiment is based on GPT-2-125M, with GPT-2-1.5B as the teacher, using Dolly as the test set. For each prompt, we sample 10 responses to reduce the variance. We can see that the ExAccErrs of the baselines continuously grow during generation, while MiniLLM has a much lower ExAccErr, and the error stops accumulating in long-text generation (
>
 150 tokens).

Figure 6:The excess error caused by the training-decoding discrepancy (ExAccErr) accumulated with the generation length. Lower ExAccErr means the method introduces less exposure bias.
Refer to caption
SST2	BoolQ
ECE	Acc.	ECE	Acc.
Teacher	0.025	93.0	0.356	74.5
KD	0.191	84.7	0.682	63.5
SeqKD	0.243	66.5	0.681	62.8
MiniLLM	0.099	89.7	0.502	67.8 
Figure 6:The excess error caused by the training-decoding discrepancy (ExAccErr) accumulated with the generation length. Lower ExAccErr means the method introduces less exposure bias.
Table 2:The ECE and accuracy scores on SST2 and BoolQ datasets. The best scores among student models are boldfaced.
Calibration
[50] has shown that models trained with policy optimization are likely to be poorly calibrated. We test the calibration of MiniLLM and the KD baselines on two widely-used text classification datasets: SST2 [61] and BoolQ [14], based on LLaMA-7B. We design zero-shot classification instructions (see Appendix B.2) and take the probability of the label words to compute the ECE scores [48]. From Table 6, we observe that KD and SeqKD models are worse calibrated than the teacher model, which potentially explains their low performance on canonical benchmarks [22]. We suspect that minimizing forward KLD causes the models to push high probabilities to void regions of the target distribution, which leads to significant distribution differences between the student and the teacher (see the example in Figure 2). In contrast, MiniLLM focuses on accurately learning the major parts of the target distribution and narrows the ECE scores gap between the student and the teacher.

Figure 7:The Rouge-L scores of the distilled models against SFT on the different subsets of S-NI split by the golden responses’ length.
Refer to caption
DollyEval	SelfInst
Dist-4	Loss	Dist-4	Loss
Teacher	99.3	3.55	99.1	4.44
SFT	99.5	3.89	99.0	5.28
MiniLLM	99.0	3.95	98.6	5.33 
Figure 7:The Rouge-L scores of the distilled models against SFT on the different subsets of S-NI split by the golden responses’ length.
Table 3:The distinct 4-grams (Dist-4) and language modeling loss (Loss) on the test sets based on the LLaMA family. MiniLLM preserves generation diversity.
Performance on Different Response Length
We study the models’ performance when the golden response lengths belong to different ranges. In Figure 7, we illustrate the Rouge-L scores of different KD models against the SFT models on three S-NI subsets split by the length of the ground truth responses. We can see that all methods achieve low scores on prompts that expect short responses (
≤
5
 tokens), probably because most responses in our training set are long sentences, introducing a distribution shift between training and evaluation [53]. Furthermore, the output spaces of these prompts are relatively small, allowing the student model to cover most modes of the teacher, and thus reverse KLD and forward KLD have similar performance. For prompts with longer responses (
≥
6
 tokens), the teacher distribution contains more modes than the students due to the complex output spaces, which shows the advantage of MiniLLM against standard KD models. Similar results on UnNI are shown in Appendix C.3.

Generation Diversity
[10] has found that the model optimized by minimizing reverse KLD is likely to lose modes, which affects the generation diversity. We follow [52] to discuss generation diversity from three aspects: (i) generating multiple distinct responses given a prompt. (ii) generating linguistically complex responses. (iii) the ability to generate contents that have high coverage of the real data distribution. For (i), we argue that for many NLP applications, generating one correct response is sufficient, especially for those scenarios demanding high truthfulness and reliability [33]. For (ii) and (iii), we report the responses’ distinct 4-gram proportion and the language modeling loss on the test sets in Table 7, using the base models from the LLaMA family (see Appendix B.4 for more details) . We can see that MiniLLM preserves the distinct 4-gram proportion in the generated responses and language modeling loss on the test set.

3.4Ablation Studies on Optimization Strategies
We evaluate the effectiveness of the three strategies proposed to stabilize and accelerate optimization in Section 2.2 by distilling a GPT-2-125M model from the GPT-2-1.5B model. More ablation studies can be found in Appendix C.4. In Table 8, we report the best Rouge-L scores on the validation set of each run and the evaluation results of the corresponding checkpoints. We also plot the reverse KLD between the student and the teacher during training in Figure 8, where the curves are smoothed by 32 steps. We can see that Teacher-Mixed Sampling and Length Normalization works for stabilizing training. Although the reverse KLDs also decrease without these strategies, we find that the models quickly learn to generate repeated, short, or meaningless strings that have high probabilities in the teacher distribution (see examples in Appendix D), which is known as reward hacking [59]. This also leads to the low generation performance in Table 8. From Figure 8, we also observe that the Single-Step Decomposition effectively reduces the variance of the training process, which also results in higher scores on the validation and test sets.

Table 4:The performance on the validation and test set when different combinations of MiniLLM optimization strategies are applied.
Valid.	Dolly
R-L	R-L
MiniLLM	27.4	24.6
  w/o Length Norm.	17.4	14.7
  w/o Teacher-Mixed	22.3	20.4
  w/o Single-Step	27.0	23.7 
Refer to caption
Table 4:The performance on the validation and test set when different combinations of MiniLLM optimization strategies are applied.
Figure 8:The reverse KLD between the teacher and the students during MiniLLM training when different optimization strategies are applied.
4Related Work
Large Language Models
Large language models (LLMs; 7, 66, 16, 50, 1) have shown superior performance by solving various NLP tasks in a generative manner. Recent works apply instruction tuning [71, 65, 12] or learning from human feedback [51, 5] to improve the alignment of LLMs with humans further and create general AI assistants [49, 21]. There are also efforts to build open-source LLMs [83, 68, 8] to facilitate research and industry development. Although appealing, the broad capacities of LLMs usually emerge with large model sizes [35, 77] that require massive computational resources. Therefore, model compression is critical for the practical deployment and further research of LLMs.

Knowledge Distillation
Knowledge distillation (KD; 28), as a widely used model compression technique, aims at training a student model with the guidance of a teacher model [55, 58, 30]. In the NLP community, many works apply KD to text classification tasks by mimicking the teacher model’s output distribution [63, 38, 84], hidden states [34, 57], or attention scores [78, 70]. For text generation, the standard KD method is to approximately minimize the forward KLD between the student’s and the teacher’s generation distribution by using the teacher’s output at each time step as supervision [58] or direct training on the teacher’s generated texts [36, 67, 15, 53]. In this paper, we minimize the reverse KLD, which is more suitable for LLMs when the teacher distribution is available. Concurrent works [3, 75] also explore more the distribution discrepancy metrics in KD.

Distribution Discrepancy Metrics in Text Generation
The distribution discrepancy metrics play a significant role in training text generation models. The forward KLD is widely used due to its simplicity when derived as the Maximum Likelihood Estimate (MLE) objective [85]. However, previous works show that minimizing forward KLD leads to zero-forcing behavior where models try to cover all modes of the target distribution and sacrifice the accuracy of major modes [27]. Some works resort to using other metrics to remedy this problem, such as reverse KLD [31], Total Variation Distance [32], and Optimal Transport [41]. Our paper tackles this problem under the scenario of knowledge distillation for LLMs.

5Conclusion
In this work, we investigate the problem of distilling the knowledge of LLMs into small language models. We find that the standard distillation methods minimizing the forward KLD is sub-optimal in language generation scenarios because the teacher’s output distribution contains more modes than the student’s, and forward KLD forces the student distribution to overestimate the low-probability regions of the teacher distribution. Therefore, we propose MiniLLM that minimizes the reverse KLD between the teacher and student distribution and design an algorithm to optimize this objective. Extensive experiments show that MiniLLM produce more precise responses that have higher overall quality than standard KD models. We also find that MiniLLM has lower exposure bias, better calibration, and higher performance in long-text generation with good generation diversity.

Acknowledgements
This work was supported by the National Key Research and Development Program of China (No. 2021ZD0113304), the National Science Foundation for Distinguished Young Scholars (with No. 62125604), and the NSFC projects (Key project with No. 61936010).
