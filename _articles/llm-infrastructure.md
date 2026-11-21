---
title: "A Stroll Through LLMs (II) — Infrastructure"
category: "Machine Learning"
order: 5
description: "A FLOP and byte budget for training parallelism, serving, MLA, and speculative decoding."
---

The block from [Part I]({{ '/writing/llm-architecture/' | relative_url }}) does the same matrix multiplications during training and generation. What changes is how many tokens can use each weight at once, what has to stay in memory, and what has to cross a device boundary. Those differences explain much of the hardware layout.

* TOC
{:toc}

## What the block costs

<figure markdown="1" style="margin-inline: 0">

![The same decoder block in both parts. It shows the attention projections, a gated MLP, and two residual additions.]({{ '/assets/llm-stroll/decoder.png' | relative_url }}){: width="725" height="722" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>The same decoder block in both parts. It shows the attention projections, a gated MLP, and two residual additions.</small></figcaption>

</figure>

### Parameters, FLOPs, and bytes

Let $$B$$ be the batch size, $$T$$ the training sequence length, and $$L$$ the number of layers. The diagram calls the model and MLP widths $$D$$ and $$F$$. Here they are $$d$$ and $$f$$. Its head counts $$N,K$$ and head width $$H$$ are $$H_q,H_{\rm kv}$$ and $$d_h$$ below. The four top matrices are the attention projections and the bottom three are the gated MLP.

$$
N_{\rm layer}=2d\,d_h(H_q+H_{\rm kv})+3df .
$$

Count a multiplication and an addition as two FLOPs. For $$Y=XW$$ with $$n$$ rows in $$X$$ and $$N_W$$ entries in $$W$$, the forward pass costs about $$2nN_W$$ FLOPs. Given the output gradient $$\bar Y$$, each of $$\bar X=\bar YW^\top$$ and $$\bar W=X^\top\bar Y$$ costs about the same. Let $$N_{\rm mat}$$ count the matrix weights used per token across the model, including the output projection. In an MoE block, count only the active experts for this compute estimate.

For causal attention let $$A_T=T(T+1)/2$$ be the valid query-key pair count. Scoring the keys and summing the values each cost about $$2d_h$$ FLOPs per pair per query head.

$$
\begin{aligned}
F_{\rm forward}(B,T)&\simeq
    2BT N_{\rm mat}+4BLH_qd_h A_T,\\
F_{\rm train}(B,T)&\simeq
    6BT N_{\rm mat}+12BLH_qd_h A_T.
\end{aligned}
$$

These are useful matmul counts for a forward or forward-backward pass, before recomputation and lower-order operations. Kernels can execute extra work on tiles that touch the causal boundary.

### The roofline

If a device must do $$F$$ FLOPs, transfer $$M_{\rm HBM}$$ bytes through its memory, and move $$M_{\rm link}$$ bytes over a link, an optimistic time bound is

$$
t\ \ge\ \max\!\left\{
 \frac{F}{P_{\rm peak}},\
 \frac{M_{\rm HBM}}{\mathrm{BW}_{\rm HBM}},\
 \frac{M_{\rm link}}{\mathrm{BW}_{\rm link}}
 \right\}.
$$

Arithmetic intensity is $$F/M_{\rm HBM}$$. Below the machine's ratio $$P_{\rm peak}/\mathrm{BW}_{\rm HBM}$$, even peak memory bandwidth cannot feed peak compute. Communication on the critical path can add time beyond this bound. Model FLOPs utilization compares the useful model FLOP rate with the aggregate hardware peak at the relevant precision.

## Training

### Data parallelism and state sharding

#### DDP and collective communication

Distributed data parallelism (DDP) gives each rank a copy of the model and a different shard of the batch. Each rank computes gradients locally. An all-reduce averages them so that the replicas take the same optimizer step. Backward can launch reductions for completed gradient buckets while it computes earlier layers.

A ring all-reduce splits an $$M$$-byte buffer into $$p$$ chunks over $$p$$ ranks. Reduce-scatter accumulates one chunk on each rank, then all-gather distributes the reduced chunks. Each phase sends one chunk per rank in each of $$p-1$$ rounds.

$$
M_{\rm sent\ per\ rank}\simeq 2\,\frac{p-1}{p}M.
$$

Receive traffic is the same size. Gradient accumulation runs several microbatches locally before reducing, so a group of $$p_{\rm DP}$$ replicas can process $$B_{\rm global}=p_{\rm DP}mB_{\rm micro}$$ examples per step with one gradient synchronization.

#### ZeRO and FSDP

DDP repeats three large pieces of persistent state. ZeRO successively shards the optimizer states, the gradients, and then the parameter weights. The table shows the approximate per-rank cost under even sharding, before activations and temporary buffers.

| Mode | Weights | Gradients | Optimizer state |
|---|---|---|---|
| DDP | $$M_w$$ | $$M_g$$ | $$M_o$$ |
| ZeRO-1 | $$M_w$$ | $$M_g$$ | $$M_o/p$$ |
| ZeRO-2 | $$M_w$$ | $$M_g/p$$ | $$M_o/p$$ |
| ZeRO-3 / full sharding | $$M_w/p$$ | $$M_g/p$$ | $$M_o/p$$ |

Fully Sharded Data Parallel (FSDP) all-gathers the next block's weights when it needs them. If it releases the full weights after forward, it gathers them again for backward. The resulting gradient is reduce-scattered directly into the shards that the optimizer updates. The largest gathered block, prefetched weights, and live activations still count toward peak memory.

<figure markdown="1" style="margin-inline: 0">

![FSDP gathers a module for local computation and reduce-scatters its gradients.]({{ '/assets/llm-stroll/fsdp.png' | relative_url }}){: width="560" height="317" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>FSDP gathers a module for local computation and reduce-scatters its gradients.</small></figcaption>

</figure>

### Tensor and sequence parallelism

#### Tensor parallelism inside the MLP

Tensor parallelism (TP) splits a large matrix multiplication across ranks. In the lower half of the diagram, give each rank some hidden channels of both up-projections and the matching rows of the down-projection. In row-major notation the block becomes

$$
Y=\sum_{r=1}^{p}
  \big[\operatorname{SiLU}(XG_r)\odot(XU_r)\big]D_r .
$$

The gate is local to each channel, so each rank can finish its contribution before the forward reduction. In backward, the contributions to the gradient of the shared input must be summed. Attention can use the same pattern by assigning heads to ranks and combining their output projections. These collectives are per block, so TP benefits from a fast local interconnect.

<figure markdown="1" style="margin-inline: 0">

![Megatron's complementary matrix splits for the MLP and self-attention.]({{ '/assets/llm-stroll/tp.png' | relative_url }}){: width="500" height="537" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Megatron&#x27;s complementary matrix splits for the MLP and self-attention.</small></figcaption>

</figure>

#### Sequence parallelism around TP

Normalizations and residual additions work independently at each token position. Sequence parallelism keeps those activations sharded along the token axis between TP operations. A reduce-scatter can put each output position on its owning rank, and an all-gather reassembles the inputs for the next TP projection. Context parallelism goes further and distributes the token axis of attention itself.

### Pipeline parallelism

#### Microbatches and bubbles

Pipeline parallelism (PP) assigns consecutive layers to stages. Each microbatch sends its activation to the next stage and receives its activation gradient on the way back. With balanced stages, $$p$$ stages, and $$m$$ microbatches, GPipe's separate forward and backward sweeps take about

$$
t_{\rm GPipe}\simeq (m+p-1)(t_F+t_B),\qquad
\text{bubble fraction}\simeq\frac{p-1}{m+p-1}.
$$

In 1F1B, stages alternate forward and backward after warmup, so activations can be freed earlier. Interleaving assigns multiple virtual stages to each device to shorten idle gaps, at the cost of more boundaries that send activations.

#### Zero-bubble schedules

Look at the two backward matmuls again. $$\bar X$$ must go to the previous stage quickly. $$\bar W$$ is needed for the local optimizer update and can often wait. Zero-bubble schedules separate the two, send the input gradient along the pipeline first, and put weight-gradient work into otherwise idle slots.

<figure markdown="1" style="margin-inline: 0">

![Two zero-bubble schedules with separate input-gradient and weight-gradient work.]({{ '/assets/llm-stroll/zero-bubble.png' | relative_url }}){: width="735" height="252" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Two zero-bubble schedules with separate input-gradient and weight-gradient work.</small></figcaption>

</figure>

The delayed work keeps some activations live longer. How much of the bubble can be filled depends on the stage times, dependency order, and available memory.

### Context parallelism and Ring Attention

Shard the sequence into blocks, keep each query block on its home rank, and circulate key/value blocks around a ring. A rank updates its attention output from the block it currently holds while the next block is in transit. The causal mask still applies when the blocks meet.

For one query, let each visible block $$j$$ give a maximum score $$m_j$$, shifted sum $$\ell_j=\sum_i e^{s_i-m_j}$$, and shifted value sum $$z_j=\sum_i e^{s_i-m_j}v_i$$. Undo the different shifts when combining blocks.

$$
m=\max_j m_j,\qquad
o=\frac{\sum_j e^{m_j-m}z_j}{\sum_j e^{m_j-m}\ell_j}.
$$

This is the same softmax as scoring all visible keys together. The ring trades key/value traffic for distributed attention memory and work. Earlier causal query blocks see fewer keys, so assigning a mix of early and late positions to each rank helps balance the work.

<figure markdown="1" style="margin-inline: 0">

![Ring Attention circulates KV blocks while each device holds a block of queries.]({{ '/assets/llm-stroll/ring.png' | relative_url }}){: width="675" height="644" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Ring Attention circulates KV blocks while each device holds a block of queries.</small></figcaption>

</figure>

### Expert parallelism

#### Dispatch, compute, combine

Expert parallelism (EP) puts different experts on different ranks. The router determines destinations. A dispatch all-to-all sends token copies to the ranks that own their selected experts, grouped matrix multiplications process the local batches, and a combine all-to-all sends the outputs back for the weighted sum. The shared path can run alongside this exchange.

The MoE matmul count scales with activated experts. The network cost depends on how many assignments leave the rank and how evenly the destinations fill. Larger expert pools can give each rank less useful work per expert, making small GEMMs and the slowest rank visible in step time. Expert and attention layers can use different rank groupings, so EP can reuse ranks that act as data replicas in the attention layers.

#### What DeepSeek-V3 changed

DeepSeek-V3 limits the number of nodes a token can visit. Dispatch sends a token over InfiniBand to a GPU with the matching local rank on a destination node, then forwards it to the expert GPUs over NVLink. This lets the fast in-node link handle the local fan-out.

DualPipe pairs forward and backward chunks from pipelines progressing in both directions. Within each pair it overlaps attention and expert compute with dispatch, combine, and stage transfers. Splitting the backward work also gives the scheduler room to prioritize the input gradient. The overlap hides communication when enough independent work is available.

<figure markdown="1" style="margin-inline: 0">

![A pair of DeepSeek-V3 forward and backward chunks overlapping computation and communication.]({{ '/assets/llm-stroll/dualpipe.png' | relative_url }}){: width="745" height="117" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>A pair of DeepSeek-V3 forward and backward chunks overlapping computation and communication.</small></figcaption>

</figure>

DeepEP exposes specialized dispatch and combine kernels for this kind of exchange. Routing balance, link traffic, and overlap still have to be measured on the actual batch and topology.

### Activations, attention kernels, and precision

FlashAttention tiles the score and value products into on-chip memory and uses an online softmax like the one above. The whole score matrix need never be written to HBM. The causal pair count is the same. Checkpointing takes a related trade-off at a larger scale, keeping fewer activations and doing part of the forward pass again in backward.

<figure markdown="1" style="margin-inline: 0">

![FlashAttention tiles the calculation to reduce HBM traffic from attention intermediates.]({{ '/assets/llm-stroll/flash.png' | relative_url }}){: width="745" height="274" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>FlashAttention tiles the calculation to reduce HBM traffic from attention intermediates.</small></figcaption>

</figure>

Lower-precision matmuls change the available compute rate and the bytes each operand needs. DeepSeek-V3 uses fine-grained FP8 scaling for matrix operands and higher precision where needed for accumulation and sensitive operations. Useful FLOPs, executed FLOPs, and stored bytes are three separate quantities.

## Inference

### Prefill and decode

**Prefill** runs the prompt's positions together and builds a key/value (KV) cache. Its leading compute is the forward formula above. **Decode** adds one position per live request and reads the earlier keys and values. If $$S_i$$ is the number of visible positions for request $$i$$ at that step, then

$$
\begin{aligned}
F_{\rm decode\ step}&\simeq
    2B N_{\rm mat}+4LH_qd_h\sum_i S_i,\\
M_{\rm KV}&\simeq 2 b_{\rm kv}LH_{\rm kv}d_h\sum_i S_i .
\end{aligned}
$$

Here $$b_{\rm kv}$$ is bytes per cached element. Decode reuses each weight across the batch, but different requests generally read their own cached history. If weight reads dominate a dense projection, its intensity is roughly $$2B/b_w$$ for $$b_w$$ bytes per weight. If cache reads dominate the attention, an analogous estimate is $$2H_q/(b_{\rm kv}H_{\rm kv})$$, assuming one load per shared key/value vector. This is why batching helps weight reuse and GQA helps attention bandwidth.

### Scheduling the server

#### Continuous batching and paged KV

Requests arrive and finish at different times. Continuous batching inserts fresh requests as other requests finish, at iteration boundaries. PagedAttention gives each request a logical KV address space backed by blocks in an allocator. The blocks can live anywhere in device memory and grow with the sequence. A prefix cache can let several requests share the same completed blocks for an identical token prefix.

<figure markdown="1" style="margin-inline: 0">

![A sequence's logical KV blocks can point to noncontiguous physical blocks.]({{ '/assets/llm-stroll/paged.png' | relative_url }}){: width="500" height="256" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>A sequence&#x27;s logical KV blocks can point to noncontiguous physical blocks.</small></figcaption>

</figure>

#### Chunked prefill and disaggregation

A long prefill can stall decode iterations that share its GPU. Chunked prefill budgets prompt tokens per iteration and interleaves those chunks with decode tokens. Separate prefill and decode workers can tune each stage for its traffic, at the cost of moving the KV state between them. The added transfer time has a lower bound $$M_{\rm transferred}/\mathrm{BW}_{\rm link}$$.

<figure markdown="1" style="margin-inline: 0">

![Separate prefill and decode worker pools transfer the KV cache between them.]({{ '/assets/llm-stroll/distserve.png' | relative_url }}){: width="500" height="301" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Separate prefill and decode worker pools transfer the KV cache between them.</small></figcaption>

</figure>

Time to first token measures the wait through queuing and prefill. Time per output token measures the pace after that. Chunk sizing, batching, and worker placement affect these two latencies differently.

### Shrinking the cache

#### Multi-head latent attention

GQA reduces the number of distinct cache heads. **Multi-head latent attention (MLA)** instead compresses the joint content of keys and values into a per-token latent. Let $$h_j$$ be the hidden vector of token $$j$$ and $$c_j$$ its cached latent. For head $$h$$, the conceptual content key and value are

$$
c_j=W^{\rm DKV}h_j,\qquad
k_{j,h}^{C}=W_h^{\rm UK}c_j,\qquad
v_{j,h}^{C}=W_h^{\rm UV}c_j.
$$

Reconstructing every head's old key at every decode step would be expensive. Move the key up-projection onto the current query, and apply the value up-projection to the summed latent afterward.

$$
\begin{aligned}
(q_{t,h}^{C})^\top k_{j,h}^{C}
  &=\big[(W_h^{\rm UK})^\top q_{t,h}^{C}\big]^\top c_j,\\
\sum_j a_{tj,h}v_{j,h}^{C}
  &=W_h^{\rm UV}\left(\sum_j a_{tj,h}c_j\right).
\end{aligned}
$$

Each head still has its own attention scores. The same cached latent supplies content to all of them. Kernels can choose a different form for prefill, where projection work can be amortized over many queries.

#### Where RoPE goes

Rotating an expanded content key would insert a position-dependent operator between the query and the key up-projection. MLA keeps a separate small RoPE key $$k_j^{R}$$ shared across heads and a positional query $$q_{t,h}^{R}$$ for each head. The score adds the content and position dot products before scaling and softmax.

$$
\begin{aligned}
s_{tj,h}&=
\frac{\big[(W_h^{\rm UK})^\top q_{t,h}^{C}\big]^\top c_j
      +(q_{t,h}^{R})^\top k_j^{R}}
     {\sqrt{d_h+d_R}},\\
\text{cache bytes per token per layer}
   &=b_{\rm kv}(d_c+d_R).
\end{aligned}
$$

$$d_c$$ is the width of the cached content latent and $$d_R$$ the width of the positional key. The cache stores $$c_j$$ and $$k_j^{R}$$. The absorbed form saves cache traffic even when its latent dot product is wider than a single expanded head. Kernels such as FlashMLA implement decoding directly on the compact cache.

<figure markdown="1" style="margin-inline: 0">

![MLA caches a shared latent for the key and value content. This schematic shows the content path.]({{ '/assets/llm-stroll/mla.png' | relative_url }}){: width="745" height="224" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>MLA caches a shared latent for the key and value content. This schematic shows the content path.</small></figcaption>

</figure>

#### DeepSeek's long-context cache

V3.2 adds sparse retrieval to MLA. A small indexer selects relevant past positions for the expensive attention read. The full history can still be available to future queries. V4 changes what gets stored as well. Compressed Sparse Attention (CSA) pools nearby token states into compressed entries and selects a subset of those entries. Heavily Compressed Attention (HCA) uses stronger compression and attends across its compressed history. A local sliding window preserves a direct view of recent tokens.

<figure markdown="1" style="margin-inline: 0">

![DeepSeek-V4's CSA selects compressed cache entries and also reads a recent local window.]({{ '/assets/llm-stroll/v4-csa.png' | relative_url }}){: width="720" height="397" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>DeepSeek-V4&#x27;s CSA selects compressed cache entries and also reads a recent local window.</small></figcaption>

</figure>

With compression stride $$m$$ and a recent window of width $$w$$, CSA keeps on the order of $$S/m+w$$ main cache entries after $$S$$ tokens and reads at most $$k+w$$ for one query when it selects $$k$$ compressed entries. The indexer has its own state and reads. MLA cuts bytes per entry, sparse selection cuts the number read, and token compression cuts the number stored.

### Serving sparse models

Replicas can serve independent requests when the model fits on each worker. Tensor sharding spreads weight reads across devices and adds reductions at each step. The faster choice depends on whether memory, compute, or the link is setting latency.

DeepSeek-V3 uses separate prefill and decode deployments and different expert-parallel layouts. Prefill can use large expert batches and overlap a batch's computation with another batch's dispatch or combine. Decode is more sensitive to small-message latency and imbalance, so the deployment uses direct transfers and replicas of heavily loaded experts. Routing to replicas can spread the load while keeping the expert function the same.

### Quantized weights and cache

Quantization shrinks the bytes per element in the same accounting. One symmetric integer scheme stores a rounded value $$q$$ and a scale $$s$$ for a group of weights or cache entries.

$$
q=\operatorname{clip}\!\big(\operatorname{round}(x/s),q_{\min},q_{\max}\big),
\qquad \hat x=sq .
$$

Weight-only quantization lowers $$b_w$$. KV quantization lowers $$b_{\rm kv}$$. The kernel may unpack or dequantize a tile as it loads it. Using low-precision activations as well can change the available matmul rate. Group scales and calibration help control error, and the format pays off when the bytes saved outweigh unpacking and scale overhead.

### Speculative decoding

#### Draft and verify

A small draft model proposes several tokens. The target scores the draft prefixes in one forward pass, then a verifier decides which prefix to emit. At an accepted prefix, let $$q$$ be the draft distribution and $$p$$ the target distribution for the next token. A proposal $$x\sim q$$ is accepted with probability

$$
a(x)=\min\!\left(1,\frac{p(x)}{q(x)}\right).
$$

On rejection, sample once from the normalized positive part $$[p-q]_+$$ and stop that draft. The total probability of outputting $$x$$ is

$$
q(x)a(x)+[p(x)-q(x)]_+
 =\min(p(x),q(x))+[p(x)-q(x)]_+=p(x).
$$

The rejection probability supplies exactly the normalizer of the correction distribution. Repeating this at each accepted prefix preserves the target's sampling law. If every proposal is accepted, the verifier can emit one extra target sample. The speedup comes from amortizing weight reads and launch overhead over more emitted tokens.

<figure markdown="1" style="margin-inline: 0">

![A draft-and-verify trace. Green marks draft suggestions, red marks rejections, and blue marks target samples.]({{ '/assets/llm-stroll/spec.png' | relative_url }}){: width="755" height="232" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>A draft-and-verify trace. Green marks draft suggestions, red marks rejections, and blue marks target samples.</small></figcaption>

</figure>

#### Learned drafters and multi-token prediction

A standalone draft is one option. EAGLE-3 uses features from multiple depths of the target model to feed a learned draft. Tree verification can batch candidate branches, with a mask that lets each branch see its own prefix. DeepSeek's multi-token prediction (MTP) trains an auxiliary block to predict a further token conditioned on the intervening token. The auxiliary block can serve as a draft during inference.

<figure markdown="1" style="margin-inline: 0">

![EAGLE-3 learns from its own draft inputs during training. Earlier EAGLE designs are shown above it.]({{ '/assets/llm-stroll/eagle3.png' | relative_url }}){: width="555" height="773" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>EAGLE-3 learns from its own draft inputs during training. Earlier EAGLE designs are shown above it.</small></figcaption>

</figure>

Parallel drafts can also save time on the draft side. DFlash conditions a block denoiser on target-model features and proposes several positions in one pass. The target still verifies a causal prefix.

Drafting pays off when the cost of proposing and verifying a batch of candidates, divided by the expected number of committed tokens, is below the cost of stepping the target once per token. Long and poorly matched drafts can spend more time evaluating candidates that never reach the output.
