---
title: "A Stroll Through LLMs (II) — Infrastructure"
category: "Machine Learning"
order: 5
description: "A symbolic account of Transformer training traffic, inference FLOPs, KV storage, and speculative decoding."
---

Follow the residual stream through the decoder block from [Part I]({{ '/writing/llm-architecture/' | relative_url }}). It enters attention, passes through the MLP, and leaves for the next block. Depending on how the model is split, a rank may need someone else's keys, a sum of partial outputs, or the result from a remote expert. Those messages scale in different ways.

* TOC
{:toc}

## What the block costs

<figure markdown="1" style="margin-inline: 0">

![The same decoder block in both parts. It shows the attention projections, a gated MLP, and two residual additions.]({{ '/assets/llm-stroll/decoder.png' | relative_url }}){: width="725" height="722" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>The same decoder block in both parts. It shows the attention projections, a gated MLP, and two residual additions.</small></figcaption>

</figure>

### Parameters, FLOPs, and bytes

Let $$B$$ be the number of sequences presented to the block, $$T$$ their length, $$n=BT$$ the number of token rows, and $$L$$ the model depth. The diagram calls the model and MLP widths $$D$$ and $$F$$. Here they are $$d$$ and $$f$$. Its head counts $$N,K$$ and head width $$H$$ are $$H_q,H_{\rm kv}$$ and $$d_h$$ below.

Using row-major token activations, the normalized input $$X\in\mathbb R^{n\times d}$$ produces $$Q=XW_Q$$, $$K=XW_K$$, and $$V=XW_V$$. The query width is $$H_qd_h$$ and each of the key and value widths is $$H_{\rm kv}d_h$$. The output projection maps the concatenated head outputs back to width $$d$$. The lower half of the block has a gate, a value projection, and a down-projection.

$$
N_{\rm block}=2d\,d_h(H_q+H_{\rm kv})+3df .
$$

For a weight matrix with $$N_W$$ entries, multiplying it by $$n$$ rows costs about $$2nN_W$$ FLOPs. The input-gradient and weight-gradient matmuls each cost about the same. Write $$N_{\rm mat}$$ for the number of matrix weights used per token across the model, averaged over token rows when the active matrices vary. This counts a routed expert on its selected rows and the output projection only on rows for which logits are computed. A causal query at position $$t$$ reads $$t$$ keys, so the pair count per sequence is $$A_T=T(T+1)/2$$. Each pair costs about $$2d_h$$ FLOPs to score and $$2d_h$$ to accumulate its value, per query head.

$$
\begin{aligned}
F_{\rm forward}(B,T)&\simeq 2nN_{\rm mat}+4BLH_qd_h A_T,\\
F_{\rm train}(B,T)&\simeq 6nN_{\rm mat}+12BLH_qd_h A_T .
\end{aligned}
$$

These count the leading matmuls under the stated mask. Recomputed tiles, padding, normalization, and the optimizer add work. We use $$b_a$$ bytes per communicated activation and $$b_g$$ per communicated activation gradient. The inference sections use $$b_w$$ for a stored weight and $$b_{\rm kv}$$ for a stored cache element.

### The roofline

For one device, let $$F$$ count its work and $$M_{\rm HBM},M_{\rm link}$$ count the bytes on its memory and a relevant network link. Independent work can overlap, so the optimistic bound is

$$
t\ \ge\ \max\!\left\{
 \frac{F}{P_{\rm peak}},\
 \frac{M_{\rm HBM}}{\mathrm{BW}_{\rm HBM}},\
 \frac{M_{\rm link}}{\mathrm{BW}_{\rm link}}
 \right\}.
$$

The useful intensity is $$F/M_{\rm HBM}$$. Compare it with $$P_{\rm peak}/\mathrm{BW}_{\rm HBM}$$ to find the compute or memory ceiling. A message on the critical path also costs startup latency. A simple model for one link transfer is $$t_{\rm msg}(M)\simeq \alpha_{\rm link}+M/\mathrm{BW}_{\rm link}$$. Repeated small collectives can therefore matter even when their total byte count is modest.

## Training

Every network volume below counts **bytes sent**, counting each transmitted copy once. Per-rank and group totals are labeled separately. The formulas count large tensor payloads and assume evenly sized shards where division is used. Routing metadata, padding, retransmission, and topology can add costs.

### Data parallelism and state sharding

#### DDP and collective communication

Distributed data parallelism (DDP) gives each of $$p_d$$ ranks a model replica and a shard of the batch. All-reduce averages their parameter gradients before the optimizer step. Completed gradient buckets can leave while backward continues, and accumulating microbatches locally lets one synchronization cover the accumulated gradients.

A ring collective splits an $$M$$-byte full tensor into $$p$$ chunks. In reduce-scatter, each rank sends $$M/p$$ bytes for each of $$p-1$$ rounds, accumulating one reduced chunk locally. All-gather uses another $$p-1$$ rounds to distribute the chunks.

$$
C_{\rm RS}(p,M)=C_{\rm AG}(p,M)=\frac{p-1}{p}M,\qquad
C_{\rm AR}(p,M)=2\frac{p-1}{p}M .
$$

These are per-rank sent bytes for the ring model. A balanced ring receives the same amount. DDP therefore sends $$C_{\rm AR}(p_d,M_{\nabla W})$$ per rank for a full parameter-gradient buffer of $$M_{\nabla W}$$ bytes.

#### ZeRO and FSDP

Grouping states by what survives the step gives three sizes $$M_W$$ for weights, $$M_{\nabla W}$$ for weight gradients, and $$M_{\rm opt}$$ for optimizer state. The table shows the persistent per-rank share before activations and temporary buffers.

| Mode | Weights | Weight gradients | Optimizer state |
|---|---|---|---|
| DDP | $$M_W$$ | $$M_{\nabla W}$$ | $$M_{\rm opt}$$ |
| ZeRO-1 | $$M_W$$ | $$M_{\nabla W}$$ | $$M_{\rm opt}/p_d$$ |
| ZeRO-2 | $$M_W$$ | $$M_{\nabla W}/p_d$$ | $$M_{\rm opt}/p_d$$ |
| ZeRO-3 / full sharding | $$M_W/p_d$$ | $$M_{\nabla W}/p_d$$ | $$M_{\rm opt}/p_d$$ |

For a fully sharded block, all-gather materializes its weights on each rank. If the full block is released after forward, backward gathers it again, then reduce-scatters the weight gradients. The per-rank volume for that block is

$$
C_{\rm FSDP}=\frac{p_d-1}{p_d}\left(2M_W+M_{\nabla W}\right).
$$

Keeping the gathered weights through backward saves one all-gather and keeps those bytes live longer. Peak memory includes gathered blocks, prefetched weights, and unfinished activations.

<figure markdown="1" style="margin-inline: 0">

![FSDP gathers a module for local computation and reduce-scatters its gradients.]({{ '/assets/llm-stroll/fsdp.png' | relative_url }}){: width="560" height="317" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>FSDP gathers a module for local computation and reduce-scatters its gradients.</small></figcaption>

</figure>

### Tensor and sequence parallelism

#### Tensor parallelism inside the MLP

On $$p_t$$ tensor-parallel ranks, every rank sees the same input rows. Split the two up-projections by output channel and give rank $$r$$ the corresponding rows of the down-projection. The partial outputs are full width in the residual dimension.

$$
Z_r=\operatorname{SiLU}(XW_{g,r})\odot(XW_{u,r}),\qquad
Y_r=Z_rW_{d,r},\qquad Y=\sum_r Y_r .
$$

**Forward.** The gate and the two up-projections stay local to the shards. Each rank computes $$Y_r\in\mathbb R^{n\times d}$$, then one all-reduce supplies the complete MLP output before the next residual addition. **Backward.** Every rank receives the same $$\bar Y$$ as the gradient of its partial output. Its two up-projections produce a partial input gradient $$\bar X_r$$. Another all-reduce adds those contributions. The gradients of the sharded weights are computed on their owners.

Attention follows the same pattern. When attention groups divide evenly, each rank projects its own query and KV heads, computes their attention, then applies its row shard of $$W_O$$. Reduce the partial width-$$d$$ outputs in forward and the partial QKV input gradients in backward. When there are too few KV heads for the TP degree, the layout must account for replication or exchange of those heads.

<figure markdown="1" style="margin-inline: 0">

![Megatron's complementary matrix splits for the MLP and self-attention.]({{ '/assets/llm-stroll/tp.png' | relative_url }}){: width="500" height="537" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Megatron&#x27;s complementary matrix splits for the MLP and self-attention.</small></figcaption>

</figure>

For the attention and MLP together, there are two forward sums and two backward sums on shape $$n\times d$$. The ring volume for one block on one rank is

$$
\begin{aligned}
C^{F}_{\rm TP}&=2C_{\rm AR}(p_t,b_a nd),\\
C^{B}_{\rm TP}&=2C_{\rm AR}(p_t,b_g nd),\\
C_{\rm TP}&=4\frac{p_t-1}{p_t}\,nd(b_a+b_g).
\end{aligned}
$$

The MLP's local forward work is about $$6ndf/p_t$$. Per forward-reduction byte, that gives $$3f/[b_a(p_t-1)]$$ FLOPs when $$p_t>1$$. As the TP group grows, each rank computes less while its reduction still spans the full hidden width. This makes fast local links and collective startup time important.

#### Sequence parallelism around TP

A residual addition or normalization can run on just the local token rows. Sequence parallelism replaces the row-projection all-reduce with a reduce-scatter so those rows stay split. It all-gathers the input rows for the next column projection. Backward reverses the roles of all-gather and reduce-scatter.

For the same full tensor size, $$C_{\rm RS}+C_{\rm AG}=C_{\rm AR}$$ in the ring model. The saving is in the live residual and normalization activations between projections, which can hold $$nd/p_t$$ elements per rank. The gathering and scattering also change where the communication sits in the schedule.

### Pipeline parallelism

#### Microbatches and bubbles

A pipeline assigns consecutive blocks to $$p_p$$ stages. After the last block on one stage, the residual tensor travels to the next stage. Its activation gradient comes back during backward. For $$m$$ microbatches with $$n_\mu=B_\mu T$$ token rows each, the sent payloads at one cut are

$$
\begin{aligned}
C^F_{\rm cut}&=m b_a n_\mu d,\qquad C^B_{\rm cut}=m b_g n_\mu d,\\
C_{\rm PP,group}&=(p_p-1)m n_\mu d(b_a+b_g).
\end{aligned}
$$

The first line counts both directions of a single cut separately. The second sums the actual transmissions across the whole pipeline. One forward transfer has time about $$\alpha_{\rm link}+b_a n_\mu d/\mathrm{BW}_{\rm link}$$ in the simple link model. The preceding or following stage waits if the needed gradient or activation arrives late. When TP or CP also shards a boundary, use the actual shard sizes sent across that cut and count any duplicate copies.

With equal stage times $$t_F,t_B$$ and separate forward and backward sweeps, the pipeline executes $$m$$ useful slots in each sweep and spends $$p_p-1$$ slots filling or draining.

$$
t_{\rm GPipe}\simeq(m+p_p-1)(t_F+t_B),\qquad
\text{bubble fraction}\simeq\frac{p_p-1}{m+p_p-1}.
$$

This idealization isolates scheduling. A slow stage or a late activation transfer can hold up the next stage. 1F1B alternates forward and backward after warmup so each stage can release activations sooner. Interleaving assigns more virtual stages to a device and uses smaller scheduled chunks. Its traffic follows the actual number of boundaries crossed.

#### Zero-bubble schedules

For $$Y=XW$$, backward has two jobs.

$$
\bar X=\bar YW^\top,\qquad
\bar W=X^\top\bar Y .
$$

The preceding stage can start its backward work as soon as $$\bar X$$ arrives. The local optimizer needs $$\bar W$$ by the update. Splitting these jobs lets the scheduler send $$\bar X$$ first and place weight-gradient work into slots that would otherwise be idle. The payload at each cut follows the same formula. Filling the slots depends on the relative compute times, the link schedule, and how long the saved $$X$$ and $$\bar Y$$ can stay live.

<figure markdown="1" style="margin-inline: 0">

![Two zero-bubble schedules with separate input-gradient and weight-gradient work.]({{ '/assets/llm-stroll/zero-bubble.png' | relative_url }}){: width="735" height="252" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Two zero-bubble schedules with separate input-gradient and weight-gradient work.</small></figcaption>

</figure>

### Context parallelism and Ring Attention

#### Forward and backward traffic

For a CP group of $$p_c$$ ranks let $$s=T/p_c$$. Each rank holds $$s$$ positions per sequence and stores $$K,V$$ with shape $$B\times s\times H_{\rm kv}\times d_h$$ each. The two payload sizes for a local KV block are

$$
M_a^{\rm KV}=2b_a B s H_{\rm kv}d_h,\qquad
M_g^{\rm KV}=2b_g B s H_{\rm kv}d_h .
$$

**Forward.** Keep queries on their owner. Process the local keys and values, then circulate each KV block to the other $$p_c-1$$ ranks. The link carries keys and values. The query rank accumulates its output and softmax statistics locally.

**Backward.** Each query rank reuses its forward softmax statistics, revisits the source KV blocks, and accumulates its $$\bar Q$$ locally. Each visiting rank produces a partial $$\bar K,\bar V$$ for the block owner. In a streaming schedule, the owner keeps its own partial, then sends its KV block without a gradient on the first hop. The rest of the ring accumulates the remote partials. The last visitor sends just the accumulated gradient back to the owner, which adds its local partial.

$$
\begin{aligned}
C^F_{\rm CP}&=(p_c-1)M_a^{\rm KV},\\
C^B_{\rm CP}&=(p_c-1)(M_a^{\rm KV}+M_g^{\rm KV}).
\end{aligned}
$$

This counts sent bytes per rank, per layer, for that hop schedule. Each KV block makes $$p_c-1$$ trips in each pass. Its remote gradient packet starts after the first visit and makes $$p_c-1$$ trips including its return. A ring that rotates every payload on every turn can make an extra trip per payload. All-gather with a backward regather and gradient reduce-scatter provides the same logical volumes as the displayed schedule, with different peak residency.

<figure markdown="1" style="margin-inline: 0">

![Ring Attention circulates KV blocks while each device holds a block of queries.]({{ '/assets/llm-stroll/ring.png' | relative_url }}){: width="675" height="644" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Ring Attention circulates KV blocks while each device holds a block of queries.</small></figcaption>

</figure>

#### Why the softmax still works

For one query, each visible block $$j$$ gives its maximum score $$m_j$$, shifted sum $$\ell_j$$, and shifted value sum $$z_j$$. Rescale them to a common maximum before combining.

$$
\begin{aligned}
\ell_j&=\sum_{i\in j}e^{s_i-m_j},
& z_j&=\sum_{i\in j}e^{s_i-m_j}v_i,\\
m&=\max_j m_j,
& o&=\frac{\sum_j e^{m_j-m}z_j}{\sum_j e^{m_j-m}\ell_j}.
\end{aligned}
$$

A fully visible query block against a KV block costs about $$4B H_qd_h s^2$$ FLOPs. At effective compute rate $$P_{\rm eff}$$, hiding the next hop behind it requires roughly

$$
\frac{4B H_qd_h s^2}{P_{\rm eff}}
 \ \gtrsim\ \alpha_{\rm link}+\frac{M_a^{\rm KV}}{\mathrm{BW}_{\rm link}} .
$$

The tile work falls quadratically with $$s$$ while the payload falls linearly, so pushing CP too far exposes links and startup latency. Causal masks leave earlier query blocks with less work. A layout that assigns both early and late positions to each rank balances this work more evenly.

### Expert parallelism

#### Dispatch, compute, combine

At the MLP box, replace the one dense MLP with a pool of $$E$$ experts. Each token picks $$k_e$$ routed experts. A shared expert path can be evaluated locally in parallel. The routed activations have width $$d_e=d$$ for an ordinary MoE. A latent MoE can project down before dispatch and use $$d_e<d$$.

Let $$\mathcal T_r$$ be the tokens entering rank $$r$$, $$n_{\rm EP}=\sum_r\lvert\mathcal T_r\rvert$$ the token rows across its EP group of $$p_e$$ ranks, and $$o(e)$$ the rank that owns expert $$e$$. A token selects the set $$\mathcal K_t$$ with mixing weights $$g_{te}$$. Count **assignments**, because one token can be sent to several destinations.

$$
D_{rs}=\sum_{t\in\mathcal T_r}\sum_{e\in\mathcal K_t}
          \mathbf 1\{o(e)=s\},\qquad
A_r=\sum_{s\ne r}D_{rs},\qquad I_r=\sum_{s\ne r}D_{sr}.
$$

The matrix $$D$$ is the dispatch plan. Its diagonal stays on the originating rank. $$A_r$$ counts remote assignments leaving rank $$r$$, and $$I_r$$ counts remote assignments coming in. The whole matrix sums to $$n_{\rm EP}k_e$$.

1. **Route and pack.** The router scores experts on each origin rank. The dispatcher groups selected rows by destination rank and expert. A variable-size dispatch exchanges counts or uses reserved receive buffers. The permutation records where each row must return.
2. **Dispatch and compute.** An all-to-all sends the packed token rows. The owner regroups rows by local expert and runs the expert MLPs as grouped GEMMs. The GEMM for one expert uses just its assigned rows.
3. **Combine.** A reverse all-to-all returns the expert outputs to the token owner. The origin undoes the permutation and adds $$g_{te}E_e(x_t)$$ over the selected experts, plus the shared path. The result goes back into the residual stream.

The routing plan usually stays available to backward. Metadata such as token indices, counts, and weights is small relative to long activation rows, and its handling can dominate short exchanges.

#### Backward and the four exchanges

Hold the selected indices fixed for this backward pass. The expert output $$y_{te}=E_e(x_t)$$ receives gradient $$g_{te}\bar y_t$$. The gate gets the scalar $$\bar y_t^\top y_{te}$$, and the input receives the sum of partial expert input gradients. The weight gradients are computed where each expert ran. If multiple groups replicate the same expert, a separate data-parallel reduction synchronizes those weight gradients.

In a per-assignment dispatcher with the mixing on the origin rank, the network steps are as follows.

| Pass | Payload sent by rank $$r$$ | Sent bytes |
|---|---|---|
| Forward dispatch | Its token copies to expert owners | $$b_a d_e A_r$$ |
| Forward combine | Its expert outputs to token owners | $$b_a d_e I_r$$ |
| Backward through combine | Weighted output gradients to experts | $$b_g d_e A_r$$ |
| Backward through dispatch | Partial input gradients to token owners | $$b_g d_e I_r$$ |

The first backward exchange reverses combine. The second reverses dispatch and sums the returned contributions on the token owner. Gate gradients can use saved expert outputs or recomputed ones. The table counts the large activation and gradient payloads.

With $$D_{\rm remote}=\sum_{r\ne s}D_{rs}$$ and $$\rho=D_{\rm remote}/(n_{\rm EP}k_e)$$, summing these sends gives

$$
\begin{aligned}
C_{\mathrm{EP},r}&=d_e(b_a+b_g)(A_r+I_r),\\
C_{\rm EP,group}&=2d_e(b_a+b_g)\rho n_{\rm EP}k_e .
\end{aligned}
$$

An evenly loaded rank sends about the group volume divided by $$p_e$$. A hot rank can have a large $$I_r$$ even when each origin sends the same number of tokens. With outbound and inbound interface rates $$\mathrm{BW}_r^{\rm out}$$ and $$\mathrm{BW}_r^{\rm in}$$, the forward dispatch alone takes at least

$$
t_{\rm dispatch}\ \ge\ b_a d_e
 \max_r\left\{\frac{A_r}{\mathrm{BW}_r^{\rm out}},
              \frac{I_r}{\mathrm{BW}_r^{\rm in}}\right\}.
$$

Shared links impose their own limits. Evenly sharded experts with uniformly spread routing send an expected fraction $$1-1/p_e$$ of assignments remotely. Increasing the EP degree can therefore reduce local weight storage while raising the share of routes that use the network.

#### Capacity, FLOPs, and the slowest rank

For a routed SwiGLU expert of inner width $$f_e$$, its weights occupy $$3d_e f_e$$ elements. If $$\ell_e$$ rows reach expert $$e$$ and $$\ell_r=\sum_{e:o(e)=r}\ell_e$$ rows reach rank $$r$$, then the routed MLP work is

$$
\begin{aligned}
F^F_{\mathrm{expert},r}&\simeq 6d_e f_e\,\ell_r,\qquad
F^{F+B}_{\mathrm{expert},r}\simeq18d_e f_e\,\ell_r,\\
F^{F+B}_{\rm experts,group}&\simeq18n_{\rm EP}k_e d_e f_e .
\end{aligned}
$$

Here $$F+B$$ counts the forward and both backward matrix multiplications. A dense router projecting width $$d$$ to $$E$$ logits separately costs about $$2n_{\rm EP}dE$$ FLOPs in forward plus selection. The shared path is another term. Expert parameter storage grows with $$E$$, while routed MLP work per token grows with $$k_e$$. Under even placement, each rank holds $$3E d_e f_e/p_e$$ routed weight elements, with optimizer state charged separately.

The mean number of assignments per rank is $$n_{\rm EP}k_e/p_e$$. If all ranks achieved the same useful compute rate, time for the expert work would grow relative to the balanced case by the load factor

$$
\lambda_{\rm load}=\frac{\max_r\ell_r}{n_{\rm EP}k_e/p_e}.
$$

Grouped GEMMs also have to use the hardware well. Many experts with tiny batches can have low utilization at the same $$\ell_r$$. With capacity factor $$c_{\rm cap}$$, a fixed-capacity implementation can reserve $$C=\lceil c_{\rm cap}n_{\rm EP}k_e/E\rceil$$ rows per expert. If it computes all padded slots, its routed FLOP multiplier is $$EC/(n_{\rm EP}k_e)$$ relative to evaluating every real assignment once. Overflows require rerouting or dropping assignments. Dropless grouped GEMMs can follow the actual row counts, with some padding for kernel alignment.

For $$\rho>0$$, dividing routed training FLOPs by sent activation bytes gives $$9 f_e/[\rho(b_a+b_g)]$$ in this model. Increasing the expert width adds more work per transferred row. Busy experts and busy links can still slow the entire step. Router balancing, placement, and batching therefore address different parts of the time.

#### What DeepSeek-V3 changed

Sending a copy for each assignment over the slow link can be wasteful when a token's experts share a destination node. Let $$D_{\rm IB}$$ count the assignments crossing a node boundary, and let $$U_{\rm IB}$$ count distinct **token, remote destination node** pairs. For forward dispatch, sending one copy per destination node changes the InfiniBand activation volume from $$b_a d_e D_{\rm IB}$$ to $$b_a d_e U_{\rm IB}$$, plus routing metadata.

$$
U_{\rm IB}=
\sum_r\sum_{t\in\mathcal T_r}
\left\lvert\{\nu(o(e)):e\in\mathcal K_t,\ \nu(o(e))\ne\nu(r)\}\right\rvert,\qquad
\frac{C_{\rm IB,dedup}}{C_{\rm IB,assign}}=\frac{U_{\rm IB}}{D_{\rm IB}} .
$$

Here $$\nu(r)$$ is the node of rank $$r$$ and $$1-U_{\rm IB}/D_{\rm IB}$$ is the fraction of activation bytes saved when $$D_{\rm IB}>0$$. DeepSeek-V3 limits the number of destination nodes per token. With a cap of $$m_n$$ nodes, it scores each node by the sum of its strongest $$k_e/m_n$$ expert affinities and restricts expert selection to the chosen nodes. The dispatch sends one activation to a GPU on each target node with the matching local index, then forwards to the expert GPUs using NVLink.

The combine path can accumulate weighted partials on the destination node before crossing InfiniBand. A hierarchical backward can similarly send one $$\bar y_t$$ per remote node, multiply by the local gate weights on arrival, and return the sum of local expert input gradients. Individual gate gradients are scalars. When all four exchanges use this packing, the ideal cross-node activation and gradient volume becomes $$2d_e(b_a+b_g)U_{\rm IB}$$. Add metadata and in-node fan-out separately.

V3 also updates a per-expert routing bias from observed load. An overloaded expert gets a lower bias and an underused expert gets a higher one. The bias acts at selection, while the mixing weights use the original affinities. Group-limited routing controls how widely tokens fan out, even under a balanced expert load.

Dispatch, expert compute, and combine are ordered for one microbatch. **DualPipe** brings in another microbatch so that a local attention or expert GEMM can run while its neighbor's dispatch or combine is in flight. Opposite pipeline directions create more chances to do this, and separating input gradients from weight gradients gives the scheduler additional work to move. If each microbatch demands $$t_{\rm comp}$$ of compute and $$t_{\rm comm}$$ on a shared link, serialized time is their sum and the optimistic steady-state budget with independent overlap is $$\max(t_{\rm comp},t_{\rm comm})$$ per microbatch. Shared GPU resources, link congestion, and a skewed routing matrix reduce the overlap you can get. DeepEP packages the dispatch and combine operations around this kind of workload.

<figure markdown="1" style="margin-inline: 0">

![A pair of DeepSeek-V3 forward and backward chunks overlapping computation and communication.]({{ '/assets/llm-stroll/dualpipe.png' | relative_url }}){: width="745" height="117" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>A pair of DeepSeek-V3 forward and backward chunks overlapping computation and communication.</small></figcaption>

</figure>

### Putting the messages back on the block

Follow an activation through a combined TP and CP layout. Each rank projects its local rows and head shard. CP circulates KV blocks for that head shard. TP then sums partial $$W_O$$ outputs for the local rows. The MLP either repeats the TP reduction pattern or sends routed rows to expert owners and combines them. PP moves the resulting residual only at stage boundaries.

For independent TP and CP axes with evenly divisible KV heads, use $$BT/p_c$$ rows in the TP reductions and $$H_{\rm kv}/p_t$$ heads in the CP ring payload. At an EP layer, count the rows the routing group actually owns. Attention and expert layers can use different rank groups. A change of sharding between them may need an additional gather or scatter.

### Activations, attention kernels, and precision

FlashAttention tiles the score and value products into on-chip memory and uses an online softmax. The intermediate score matrix stays off HBM and the valid pair count still sets the leading work. Activation checkpointing saves other intermediates selectively and pays for recomputation in backward. Recomputing a full block once adds its forward FLOPs to the step.

<figure markdown="1" style="margin-inline: 0">

![FlashAttention tiles the calculation to reduce HBM traffic from attention intermediates.]({{ '/assets/llm-stroll/flash.png' | relative_url }}){: width="745" height="274" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>FlashAttention tiles the calculation to reduce HBM traffic from attention intermediates.</small></figcaption>

</figure>

Lower precision changes operand bytes and the rate of the available matmul kernels. DeepSeek-V3 uses fine-grained FP8 scales for matrix operands and higher precision where needed for accumulation and sensitive operations. A communication format can be chosen independently. Charge its payload bytes on the links and its conversion work on the device.

## Inference

### Prefill and decode

**Prefill** runs the prompt's positions together and writes keys and values into the cache. Its leading work is the forward formula above. **Decode** processes one fresh position per live request. Let $$S_i$$ be the number of keys visible to that new query, including its own position, and $$S_\Sigma=\sum_i S_i$$. For GQA, define the FLOPs for one query-key pair across all heads, $$\kappa_{\rm GQA}$$, and the cache bytes for one token in one layer, $$a_{\rm GQA}$$.

$$
\begin{aligned}
\kappa_{\rm GQA}&=4H_qd_h,\qquad
a_{\rm GQA}=2b_{\rm kv}H_{\rm kv}d_h,\\
F_{\rm decode}&\simeq 2BN_{\rm mat}+L\kappa_{\rm GQA} S_\Sigma,\\
M_{\rm KV}&=L a_{\rm GQA} S_\Sigma .
\end{aligned}
$$

With one ideal read of every used key and value, the attention history read has the same byte count as the live cache. Its intensity is $$\kappa_{\rm GQA}/a_{\rm GQA}=2H_q/(b_{\rm kv}H_{\rm kv})$$. Query heads that share a KV head can reuse its data. Dense projections reuse weights across the batch and have intensity roughly $$2B/b_w$$ when the weight read dominates. Growing $$B$$ helps that reuse while also growing $$S_\Sigma$$ if it adds independent requests.

### Scheduling the server

#### Continuous batching and paged KV

Continuous batching admits new requests between decode iterations as existing requests finish. PagedAttention gives each request a logical sequence of cache blocks backed by pages anywhere in device memory. With $$P$$ token slots per page, and without sharing, the allocation for uniform layers is

$$
M_{\rm alloc}=L a_{\rm GQA} P\sum_i\left\lceil S_i/P\right\rceil .
$$

Subtracting $$L a_{\rm GQA}S_\Sigma$$ counts the unfilled slots. There are at most $$P-1$$ such slots per request. A prefix cache lets multiple requests refer to the same physical blocks for an identical completed prefix. It can save both prefill work and resident bytes. Each request still has its own query when it attends to the shared keys.

<figure markdown="1" style="margin-inline: 0">

![A sequence's logical KV blocks can point to noncontiguous physical blocks.]({{ '/assets/llm-stroll/paged.png' | relative_url }}){: width="500" height="256" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>A sequence&#x27;s logical KV blocks can point to noncontiguous physical blocks.</small></figcaption>

</figure>

#### Chunked prefill and disaggregation

A long prefill can hold up decode iterations on a shared GPU. Chunked prefill budgets prompt rows per iteration and interleaves them with decode. Separate prefill and decode workers can tune each stage for its traffic, with a transfer of the prompt's cache between them. A GQA cache holding $$T$$ prompt tokens for a request contributes $$L a_{\rm GQA}T$$ bytes to that transfer before compression or sharding. The link time has a lower bound of transferred bytes divided by link bandwidth.

<figure markdown="1" style="margin-inline: 0">

![Separate prefill and decode worker pools transfer the KV cache between them.]({{ '/assets/llm-stroll/distserve.png' | relative_url }}){: width="500" height="301" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Separate prefill and decode worker pools transfer the KV cache between them.</small></figcaption>

</figure>

Time to first token covers queueing and prompt processing. Time per output token measures the pace afterward. Chunk sizing and worker placement affect the two latencies differently.

### Shrinking the cache

#### Multi-head latent attention

GQA reduces the number of independent cache heads. Multi-head latent attention (MLA) compresses the content of keys and values jointly. Let $$h_j$$ be token $$j$$'s hidden vector and $$c_j\in\mathbb R^{d_c}$$ its cached latent. For head $$h$$ the expanded content tensors would be

$$
c_j=W^{\rm DKV}h_j,\qquad
k_{j,h}^{C}=W_h^{\rm UK}c_j,\qquad
v_{j,h}^{C}=W_h^{\rm UV}c_j .
$$

Move the key up-projection onto the current query and the value up-projection after the weighted sum.

$$
\begin{aligned}
(q_{t,h}^{C})^\top k_{j,h}^{C}
   &=\big[(W_h^{\rm UK})^\top q_{t,h}^{C}\big]^\top c_j,\\
\sum_j a_{tj,h}v_{j,h}^{C}
   &=W_h^{\rm UV}\left(\sum_j a_{tj,h}c_j\right).
\end{aligned}
$$

Each head still makes its own read, with its own attention weights, from the same latent history. The moved projections run per fresh query. Expanding old per-head keys and values would run per history entry. Prefill has enough queries to make a different kernel layout attractive.

#### Where RoPE goes

A position-dependent rotation on an expanded content key would sit between its up-projection and the query. MLA instead stores a separate RoPE key $$k_j^{R}\in\mathbb R^{d_R}$$ shared across heads. Its positional query $$q_{t,h}^{R}$$ adds a second dot product to the content score.

$$
s_{tj,h}=
\frac{
  \big[(W_h^{\rm UK})^\top q_{t,h}^{C}\big]^\top c_j+
  (q_{t,h}^{R})^\top k_j^{R}
}{\sqrt{d_h+d_R}} .
$$

<figure markdown="1" style="margin-inline: 0">

![MLA caches a shared latent for the key and value content. This schematic shows the content path.]({{ '/assets/llm-stroll/mla.png' | relative_url }}){: width="745" height="224" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>MLA caches a shared latent for the key and value content. This schematic shows the content path.</small></figcaption>

</figure>

#### How much MLA saves

At the same cache precision, the entry holds $$d_c+d_R$$ values. Compare against GQA with $$H_{\rm kv}$$ KV heads at the same head width. For MHA, set $$H_{\rm kv}=H_q$$.

$$
\begin{aligned}
M_{\rm MLA}&=L S_\Sigma\,b_{\rm kv}(d_c+d_R),\\
M_{\rm GQA}-M_{\rm MLA}
   &=L S_\Sigma\,b_{\rm kv}\big(2H_{\rm kv}d_h-d_c-d_R\big),\\
\text{fraction of KV bytes saved}
   &=1-\frac{d_c+d_R}{2H_{\rm kv}d_h}.
\end{aligned}
$$

The fraction is positive when $$d_c+d_R<2H_{\rm kv}d_h$$. The baseline matters. A GQA layer with a small number of KV heads starts from a smaller cache than MHA. The same fraction describes ideal history read savings if the kernel loads each used entry once and reuses it across heads.

The absorbed score uses $$d_c+d_R$$ multiplications per head per position, and the value sum uses $$d_c$$. In the notation of the decode formula, the main history cost per pair across heads becomes

$$
\kappa_{\rm MLA}=2H_q(2d_c+d_R),\qquad
F_{\rm history,MLA}\simeq L\kappa_{\rm MLA}S_\Sigma .
$$

The difference from GQA's history work is $$2LH_q S_\Sigma(2d_c+d_R-2d_h)$$ FLOPs. The per-query projections are separate. A wider latent can raise the arithmetic cost while saving HBM traffic. The roofline decides which effect matters for the workload.

#### Sparse attention FLOPs

Write $$\kappa_{\rm core}$$ for the FLOPs per attended pair across all heads in the chosen kernel, and $$a_{\rm core}$$ for bytes per main cache entry. Sparse attention changes the pair count. A causal sliding window of width $$w$$ reads $$\min(t,w)$$ keys at position $$t$$. Putting $$\tilde w=\min(T,w)$$ gives

$$
\begin{aligned}
A_{T,w}&=\sum_{t=1}^T\min(t,w)
 =\tilde w T-\frac{\tilde w(\tilde w-1)}2,\\
F_{\rm prefill,saved}
  &\simeq BL\kappa_{\rm core}(A_T-A_{T,w}),\qquad
\frac{F_{\rm sparse,core}}{F_{\rm dense,core}}
  \simeq\frac{A_{T,w}}{A_T}.
\end{aligned}
$$

A global top-$$k_s$$ selector that keeps the same number of keys per row has that count with $$w=k_s$$. It can select a distant position and incurs its own indexing cost. For the whole forward pass, divide the saved FLOPs by $$2BT N_{\rm mat}+BL\kappa_{\rm core} A_T$$ to get the fractional saving before that extra cost. Kernel work also depends on how much masked work remains inside partially filled tiles.

For a decode batch, let $$R_i\le S_i$$ be the number of entries actually read per query. Each layer saves $$\kappa_{\rm core}\sum_i(S_i-R_i)$$ main-attention FLOPs and ideally $$a_{\rm core}\sum_i(S_i-R_i)$$ main-cache read bytes. With a pure sliding window the cache can evict expired entries. With global selection, a key that goes unread this step can still be needed by a later query.

#### DeepSeek's long-context cache

**V3.2.** DeepSeek Sparse Attention (DSA) adds a lightweight indexer to MLA. An indexer with $$H_I$$ query heads of width $$d_I$$ scores a past index key using a weighted sum of rectified dot products. Scoring every visible key costs about $$2H_Id_I S_\Sigma$$ matmul FLOPs per layer in a decode step. If the main attention reads $$R_i=\min(k_s,S_i)$$ entries, its net arithmetic saving per layer is approximately

$$
\Delta F_{\rm DSA}\simeq
\kappa_{\rm MLA}\sum_i(S_i-R_i)-2H_Id_I S_\Sigma-F_{\rm extra}.
$$

Here $$F_{\rm extra}$$ accounts for remaining arithmetic such as index projections. Top-$$k_s$$ selection also uses comparisons, scans, and memory traffic. The prefill index dots cost about $$2BH_Id_I A_T$$ FLOPs per layer. With one read per shared index key of precision $$b_I$$, the ideal history bytes saved per decode layer, before scattered-access overhead, are

$$
\Delta M_{\rm DSA}\simeq
 a_{\rm MLA}\sum_i(S_i-R_i)-b_I d_I S_\Sigma,\qquad
 a_{\rm MLA}=b_{\rm kv}(d_c+d_R).
$$

Main-cache storage can still scale with the whole history. A large main FLOP saving can coexist with a limiting index scan or scattered reads.

**V4.** Compressed Sparse Attention (CSA) stores compressed entries at stride $$m_c$$ and selects top entries from that compressed history. Heavily Compressed Attention (HCA) uses a larger stride $$m_h$$ and reads its compressed history without a top-$$k_s$$ selector. Both preserve a recent uncompressed window. For a sequence with $$S$$ visible keys, write $$W=\min(w,S)$$ for the local window size. The main entries are approximately

| Layer | Entries stored | Entries read by one query |
|---|---|---|
| CSA | $$\lfloor S/m_c\rfloor+W$$ | $$\min(k_s,\lfloor S/m_c\rfloor)+W$$ |
| HCA | $$\lfloor S/m_h\rfloor+W$$ | $$\lfloor S/m_h\rfloor+W$$ |

The table counts completed groups and the local window, apart from boundary buffers. Call the stored and read counts $$C_{\rm store}$$ and $$C_{\rm read}$$. With the same entry format and head shape as an uncompressed baseline, the fractional main memory and core FLOP savings are $$1-C_{\rm store}/S$$ and $$1-C_{\rm read}/S$$, respectively. CSA also keeps index keys and scores roughly $$S/m_c$$ compressed positions. Compression projections, boundary buffers, and index selection have their own costs.

<figure markdown="1" style="margin-inline: 0">

![DeepSeek-V4's CSA selects compressed cache entries and also reads a recent local window.]({{ '/assets/llm-stroll/v4-csa.png' | relative_url }}){: width="720" height="397" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>DeepSeek-V4&#x27;s CSA selects compressed cache entries and also reads a recent local window.</small></figcaption>

</figure>

### Serving sparse models

Let $$\mathcal U=\bigcup_i \mathcal K_i$$ be the experts touched by a decode batch of $$B$$ tokens. On a streamed-weight path that reads each touched expert's weights once, the routed MLP has

$$
M_{\rm routed\ weights}\simeq 3b_w d_e f_e\,\lvert\mathcal U\rvert,\qquad
F_{\rm routed}\simeq6Bk_e d_e f_e .
$$

Its weight-only intensity is about $$2Bk_e/(b_w\lvert\mathcal U\rvert)$$. A batch that concentrates on the same experts reuses more weights. A broad set of selected experts spreads the traffic over more weights. The per-assignment forward dispatch and combine from the training section send $$2b_a d_e\rho Bk_e$$ bytes across the EP group, before deduplication and metadata.

When the model fits, replicas can serve independent requests. Sharding spreads weight reads and can put reductions or all-to-all on every decode step. DeepSeek-V3 separates prefill and decode deployments. Prefill can use larger expert batches and overlap work from different batches. Decode favors fast direct small-message transfers and can replicate heavily loaded experts. Replicas reduce the hot destination's load in exchange for extra stored weights.

### Quantized weights and cache

Suppose a group uses a quantization step $$\Delta$$ and a bounded integer code $$q$$. Store $$q$$ and the group's scale, then approximate the original value by $$\hat x$$.

$$
q=\operatorname{clip}\big(\operatorname{round}(x/\Delta),q_{\min},q_{\max}\big),
\qquad \hat x=\Delta q.
$$

Within the unclipped range, rounding gives $$\lvert x-\hat x\rvert\le\Delta/2$$. Group scales let smaller ranges use more of the code space. They cost metadata and the kernel has to load them.

For $$N$$ elements stored at $$r$$ bits each in groups of $$G$$, let $$b_m$$ be the bytes of scale and optional zero-point per group. Ignoring alignment padding, storage and effective bytes per element are

$$
M_{\rm packed}\simeq\frac{r}{8}N+\left\lceil N/G\right\rceil b_m,\qquad
b_{\rm eff}\simeq\frac{r}{8}+\frac{b_m}{G}.
$$

Relative to $$b$$ bytes per element, the ideal storage saving is $$1-b_{\rm eff}/b$$. Put $$b_{\rm eff}$$ in place of $$b_w$$ for weights or $$b_{\rm kv}$$ for cache data. In mixed-precision MLA, the entry instead costs $$d_c b_c^{\rm eff}+d_R b_R^{\rm eff}$$ if the content and positional parts use different formats.

The leading dense multiply still has $$2nN_W$$ FLOPs for the same dimensions. Unpacking or dequantizing operands can add work. Low-precision weights and activations can raise the available matmul rate when the hardware runs that format natively. For a bandwidth-bound operation whose weight traffic dominates, the ideal time ratio is $$b_{\rm eff}/b_w$$ relative to the original. A real kernel pays for other reads, scale loads, and exposed conversion time. Once compute or another buffer dominates, those terms set the ceiling.

### Speculative decoding

#### Draft and verify

A draft proposes $$K_d$$ tokens and the target scores their causal prefixes together. At one accepted prefix, let $$q$$ be the actual draft sampling distribution and $$p$$ the target sampling distribution, including any chosen sampling transforms. The draft draws $$x\sim q$$ and the verifier accepts it with probability $$a(x)$$.

$$
a(x)=\min\!\left(1,\frac{p(x)}{q(x)}\right).
$$

A proposed token has $$q(x)>0$$. The total acceptance probability is the overlap of the two distributions.

$$
\alpha=\sum_x q(x)a(x)=\sum_x\min(p(x),q(x))
       =1-\tfrac12\sum_x\lvert p(x)-q(x)\rvert .
$$

The last term is total variation distance. A closer draft buys more accepted positions. When a proposal is rejected, replace it using the positive part of $$p-q$$, normalized by the rejection probability $$1-\alpha$$.

$$
r(x)=\frac{[p(x)-q(x)]_+}{1-\alpha},\qquad
q(x)a(x)+(1-\alpha)r(x)=\min(p(x),q(x))+[p(x)-q(x)]_+=p(x).
$$

The correction is used only when rejection has positive probability. The verifier stops the current draft on a rejection. If all proposed tokens are accepted, it draws a bonus token from the target at the accepted prefix. Every committed token thus has the target's conditional law at its actual prefix. Applying the conditional identity at each position gives the same joint sampling law as ordinary autoregressive decoding.

<figure markdown="1" style="margin-inline: 0">

![A draft-and-verify trace. Green marks draft suggestions, red marks rejections, and blue marks target samples.]({{ '/assets/llm-stroll/spec.png' | relative_url }}){: width="755" height="232" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>A draft-and-verify trace. Green marks draft suggestions, red marks rejections, and blue marks target samples.</small></figcaption>

</figure>

#### Accepted tokens and the speedup threshold

Consider a round away from end-of-sequence or length limits. Besides the accepted candidates, it commits one correction or bonus token. If $$A_j$$ means the candidate at position $$j$$ is accepted, then

$$
\begin{aligned}
\mathbb E[N]&=1+\sum_{j=1}^{K_d}\Pr(A_1,\ldots,A_j)
          =1+\sum_{j=1}^{K_d}\prod_{i=1}^{j}\alpha_i,\\
\alpha_i&=\Pr(A_i\mid A_1,\ldots,A_{i-1}).
\end{aligned}
$$

The product uses conditional probabilities. Under a constant-acceptance approximation $$\alpha_i=\alpha$$, this gives $$\mathbb E[N]=(1-\alpha^{K_d+1})/(1-\alpha)$$ for $$\alpha<1$$ and $$K_d+1$$ at $$\alpha=1$$. Later draft slots contribute only on the rounds that survive the earlier ones.

Let $$t_0$$ be the time for one ordinary target decode step at comparable load. Let $$t_d(K_d),t_v(K_d),t_o$$ be the average times for drafting, verification, and exposed overhead in a round. The throughput speedup model is

$$
\text{speedup}\simeq\frac{t_0\,\mathbb E[N]}{t_d(K_d)+t_v(K_d)+t_o}.
$$

A win requires the denominator to be below $$t_0\mathbb E[N]$$. For a fixed-size batch, use the mean expected length per request, since one ordinary step commits one token per request. In the bandwidth-bound ideal with $$t_v\simeq t_0$$ and a sequential drafter taking $$\tau_d$$ per slot, this becomes

$$
K_d\frac{\tau_d}{t_0}+\frac{t_o}{t_0}
  <\sum_{j=1}^{K_d}\alpha^j .
$$

A longer chain raises draft work with each slot, while the extra expected tokens fall with survival to that slot. Improving the proposal distribution and lowering draft cost help different sides of the inequality.

#### Verification FLOPs and weight reuse

Let $$J_v$$ be the new positions evaluated by the target in one verification pass. The exact value depends on which prefix logits and cache entries the implementation already has. For request $$i$$ with $$S_i^0$$ previously cached positions, the valid new pairs with full causal attention are $$J_v S_i^0+J_v(J_v+1)/2$$. Each fresh query sees the old prefix and a causal portion of the new block. For an equal verification block size across the batch,

$$
F_v\simeq 2BJ_v N_{\rm mat}
 +L\kappa_{\rm core}
   \left[J_v\sum_i S_i^0+\frac{B J_v(J_v+1)}{2}\right].
$$

A sequential, full-attention Transformer drafter has the same pair count for its evaluated positions, with its own $$N_{\rm mat}$$, depth, and head dimensions. Its FLOPs add across these steps even though its weights are smaller. For a single request, target and draft work per committed token is about $$(F_v+F_d)/\mathbb E[N]$$. When several requests share a verification pass, divide the batch work by the sum of their expected committed tokens. Some evaluated positions never commit, so the FLOP cost can rise. The speedup comes when the pass reuses streamed weights across several queries and amortizes the reads and launches across several committed tokens. If the target is already compute-bound, verification's extra matmuls count directly against the time budget.

#### Learned drafters and multi-token prediction

EAGLE-3 uses features from multiple depths of the target to feed a learned draft. Tree verification can batch several branches while keeping their causal histories separate. DeepSeek's multi-token prediction (MTP) trains auxiliary blocks to predict a further token conditioned on the intervening one. An auxiliary block can draft during inference.

<figure markdown="1" style="margin-inline: 0">

![EAGLE-3 learns from its own draft inputs during training. Earlier EAGLE designs are shown above it.]({{ '/assets/llm-stroll/eagle3.png' | relative_url }}){: width="555" height="773" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>EAGLE-3 learns from its own draft inputs during training. Earlier EAGLE designs are shown above it.</small></figcaption>

</figure>

DFlash uses target features to propose a block in parallel and can lower $$t_d(K_d)$$ when a sequential draft is the bottleneck. Exact stochastic verification depends on using the actual proposal distribution at each verified prefix. Changing the drafter, the verification layout, or the batch changes the quantities in the speedup test.
