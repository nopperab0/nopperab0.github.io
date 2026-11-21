---
title: "A Stroll Through LLMs (I) — Architecture"
category: "Machine Learning"
order: 4
description: "A decoder block as a guide to MoE, new residual paths, linear attention, and recurrent depth."
---

Start with one token entering a GPT-2 block. Attention lets it look back at earlier tokens. The MLP then transforms its own features. Both updates go back into the same residual stream. Most of the changes below make sense once you can point to the part of that block they change.

* TOC
{:toc}

## The decoder block

### GPT-2 as a starting point

A decoder-only language model predicts the next token from its prefix. GPT-2 starts with a sum of token and learned position embeddings, applies a stack of pre-normalized blocks, and uses a final normalization and the transposed token-embedding matrix to produce logits.

$$
\begin{aligned}
p_\theta(x_{1:T})&=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}),\\
u_\ell&=h_\ell+\operatorname{Attn}_\ell(\operatorname{LN}(h_\ell)),\\
h_{\ell+1}&=u_\ell+\operatorname{MLP}_\ell(\operatorname{LN}(u_\ell)).
\end{aligned}
$$

<figure markdown="1" style="margin-inline: 0">

![The same decoder block in both parts. It shows the attention projections, a gated MLP, and two residual additions.]({{ '/assets/llm-stroll/decoder.png' | relative_url }}){: width="725" height="722" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>The same decoder block in both parts. It shows the attention projections, a gated MLP, and two residual additions.</small></figcaption>

</figure>

The figure will serve as our map. It shows the modern form with grouped heads and a gated MLP. For GPT-2, set one query head per key/value head and use a single GELU branch with one up-projection and one down-projection. The bypasses on the left are the residual stream.

### What a head computes

For one head of width $$d_h$$, the three projections give a query $$q_t$$ at the current position and keys $$k_j$$ and values $$v_j$$ at positions $$j\le t$$. The query scores keys and those scores determine how much of each value to read. The causal mask excludes later positions.

$$
o_t=\sum_{j\le t} a_{tj}v_j,\qquad
a_{tj}=\frac{\exp(q_t^\top k_j/\sqrt{d_h})}
{\sum_{s\le t}\exp(q_t^\top k_s/\sqrt{d_h})}.
$$

The output projection mixes the head outputs back into the model width. With $$T$$ tokens, each head reads a triangular set of $$T(T+1)/2$$ query-key pairs. During generation, cached keys and values let a new query reuse that history.

## Four changes by Llama 2

Llama 2 kept the same two residual additions. It inherited RMSNorm, SwiGLU, and RoPE from Llama, and adopted grouped-query attention in its larger variants. Each choice changes a different part of the diagram.

### RMSNorm at the norm boxes

Let $$x\in\mathbb R^d$$ be one token's hidden vector. LayerNorm centers it and scales by its standard deviation. RMSNorm scales by its root mean square, with a learned gain for each coordinate.

$$
\begin{aligned}
\operatorname{LN}(x)&=g\odot
   \frac{x-\bar x\mathbf 1}{\sqrt{d^{-1}\lVert x-\bar x\mathbf 1\rVert_2^2+\epsilon}}+b,\\
\operatorname{RMSNorm}(x)&=g\odot
   \frac{x}{\sqrt{d^{-1}\lVert x\rVert_2^2+\epsilon}} .
\end{aligned}
$$

RMSNorm leaves the mean of the hidden vector free to vary. It needs just a reduction over squared coordinates and keeps the pre-norm placement that gives the skip path a direct route through the stack.

<figure markdown="1" style="margin-inline: 0">

![Validation BLEU by training step in a translation experiment with shifted initialization.]({{ '/assets/llm-stroll/rmsnorm.png' | relative_url }}){: width="420" height="288" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Validation BLEU by training step in a translation experiment with shifted initialization.</small></figcaption>

</figure>

### SwiGLU in the MLP

The two up-projections in the diagram produce a gate and a value. SwiGLU takes the SiLU of the gate and multiplies the branches coordinate by coordinate.

$$
\operatorname{SwiGLU\!-\!MLP}(x)=
  W_{\rm down}\big[\operatorname{SiLU}(W_gx)\odot W_ux\big],
  \qquad \operatorname{SiLU}(a)=a\sigma(a).
$$

A two-matrix MLP of width $$f$$ uses about $$2df$$ weights. A gated MLP of width $$f'$$ uses $$3df'$$, so choosing $$f'\approx 2f/3$$ holds that budget constant.

Kimi K3's **SiTU-GLU** smooth-caps the linear factor of the gate and the value branch. For coordinates $$a$$ and $$b$$ from the two projections, the product is bounded by $$\beta_g\beta_u$$ in magnitude. This limits large activations before low-precision operations.

$$
c_\beta(z)=\beta\tanh(z/\beta),\qquad
\operatorname{SiTU\!-\!GLU}(a,b)=
 \big[\sigma(a)c_{\beta_g}(a)\big]\,c_{\beta_u}(b).
$$

<figure markdown="1" style="margin-inline: 0">

![The scalar responses of GLU, SwiGLU, and SiTU-GLU when both branches receive the same input.]({{ '/assets/llm-stroll/situ.png' | relative_url }}){: width="510" height="303" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>The scalar responses of GLU, SwiGLU, and SiTU-GLU when both branches receive the same input.</small></figcaption>

</figure>

### RoPE on the queries and keys

GPT-2 adds an absolute position vector at the input. Rotary position embedding (RoPE) applies a position-dependent rotation to pairs of query and key coordinates. Let $$R_t$$ rotate each pair by $$t$$ times its frequency. The dot product then depends on the relative rotation.

$$
(R_tq_t)^\top(R_jk_j)
 =q_t^\top R_t^\top R_j k_j
 =q_t^\top R_{j-t}k_j .
$$

The token content still sets $$q_t$$ and $$k_j$$. The distance enters through the rotation between them. The values pass through without this rotation.

<figure markdown="1" style="margin-inline: 0">

![RoPE rotates pairs of query and key features before the dot product.]({{ '/assets/llm-stroll/rope.png' | relative_url }}){: width="565" height="331" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>RoPE rotates pairs of query and key features before the dot product.</small></figcaption>

</figure>

### GQA at the key and value projections

Multi-head attention (MHA) gives each query head its own keys and values. Grouped-query attention (GQA) lets several query heads read one shared set. Each query still gets its own scores and weighted sum.

$$
g=\frac{H_q}{H_{\rm kv}},\qquad
\text{KV elements per token per layer}=2H_{\rm kv}d_h.
$$

The query groups are the reshapes in the decoder diagram. The extremes are MHA with $$H_{\rm kv}=H_q$$ and multi-query attention (MQA) with $$H_{\rm kv}=1$$. Sharing primarily cuts the cache. The number of query heads still governs how many attention scores we compute.

<figure markdown="1" style="margin-inline: 0">

![From left to right, one KV head per query, one per group, and one for all queries.]({{ '/assets/llm-stroll/gqa.png' | relative_url }}){: width="720" height="250" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>From left to right, one KV head per query, one per group, and one for all queries.</small></figcaption>

</figure>

## The MLP becomes a mixture of experts

### Token choice and expert choice

In a dense block, every token goes through the same MLP. A mixture of experts (MoE) gives each token a choice among $$E$$ MLPs and evaluates only the selected $$k$$. In a token-choice router, logits $$r_e(x)$$ score the experts and the selected weights are normalized for the mixture.

$$
\begin{aligned}
p_e(x)&=\operatorname{softmax}(r(x))_e,\qquad
\mathcal K(x)=\operatorname{TopK}_e(p_e(x),k),\\
y(x)&=\sum_{e\in\mathcal K(x)}
  \frac{p_e(x)}{\sum_{j\in\mathcal K(x)}p_j(x)}E_e(x).
\end{aligned}
$$

Adding experts increases the parameter pool. For equal-size experts, the expert MLP work per token grows with $$k$$, and the system still has to store the full pool. In **expert choice**, each expert takes its top-scoring tokens up to a capacity. That balances slots per expert and allows the number of experts per token to vary.

### Why routers need balancing

A router can send a large share of the batch to a few experts. A capacity-limited implementation has to reroute or drop overflowing assignments. A dropless implementation still has to wait for the busiest ranks.

Let $$f_e$$ be expert $$e$$'s fraction of hard assignments and $$\bar p_e$$ its mean router probability. A common auxiliary loss sends gradient through the soft probabilities while treating the hard counts as fixed.

$$
L_{\rm balance}=\lambda E\sum_e f_e\bar p_e,\qquad
\bar p_e=\frac1N\sum_{t=1}^N p_e(x_t).
$$

Here $$N$$ counts tokens. The same imbalance matters when the router runs on one device and becomes a communication and tail-latency problem when the experts live on many.

### DeepSeekMoE and Kimi K3

DeepSeekMoE gives the layer several narrower routed experts per token and keeps a shared path for patterns many tokens need. DeepSeek-V3 uses sigmoid affinities $$s_e=\sigma(r_e)$$ and adds an expert bias for selection. It combines outputs using the unshifted affinities.

$$
\mathcal K(x)=\operatorname{TopK}_e(s_e+b_e,k),\qquad
y(x)=E_{\rm shared}(x)+\lambda_r
          \sum_{e\in\mathcal K(x)}\frac{s_e}{\sum_{j\in\mathcal K(x)}s_j}E_e(x).
$$

A busy expert's bias falls and an underused expert's bias rises. This handles the main load-balancing adjustment outside the gradient of the language-model loss. V3 also uses a small sequence-level balancing loss and restricts how widely each token is dispatched across nodes.

<figure markdown="1" style="margin-inline: 0">

![DeepSeekMoE splits routed experts into finer units and sets aside a shared expert path.]({{ '/assets/llm-stroll/deepseek-moe.png' | relative_url }}){: width="730" height="348" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>DeepSeekMoE splits routed experts into finer units and sets aside a shared expert path.</small></figcaption>

</figure>

Kimi K3 puts the routed experts in a narrower space. It dispatches $$W_\downarrow x$$, normalizes the weighted expert output, then projects it back with $$W_\uparrow$$. Shared experts still see the full-width vector.

Its quantile balancing records the $$(k+1)$$-th biased score $$c_t$$ for each token. With those cutoffs fixed, an expert enters the selection when its margin $$s_{te}-c_t$$ exceeds $$-b_e$$. Setting the threshold at the $$(1-k/E)$$-quantile of its margins targets a fraction $$k/E$$ of tokens for that expert. The bias is updated for the following batch and frozen for inference.

## The residual stream across depth

### DeepSeek's mHC

The two plus signs in the diagram add an update to one stream. Hyper-connections keep $$n$$ parallel residual streams and let a sublayer read from, write to, and mix them. Write their stack as $$X_\ell\in\mathbb R^{n\times d}$$.

$$
X_{\ell+1}
 =H_\ell X_\ell
  +a_\ell F_\ell\big(b_\ell^\top X_\ell\big),\qquad
H_\ell\ge 0,\quad H_\ell\mathbf 1=\mathbf 1,\quad
H_\ell^\top\mathbf 1=\mathbf 1 .
$$

The vectors $$b_\ell$$ and $$a_\ell$$ read from and write to the streams. Manifold-constrained Hyper-Connections (mHC) uses Sinkhorn normalization to constrain the skip mixer $$H_\ell$$ to be approximately doubly stochastic. In the exact constraint, each output is a convex combination and the mean across streams is preserved. Products of these skip matrices preserve the same property.

<figure markdown="1" style="margin-inline: 0">

![A single residual stream, unconstrained hyper-connections, and the constrained mHC mixer.]({{ '/assets/llm-stroll/mhc.png' | relative_url }}){: width="730" height="404" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>A single residual stream, unconstrained hyper-connections, and the constrained mHC mixer.</small></figcaption>

</figure>

### Kimi's Attention Residuals

Attention Residuals change what a sublayer reads. Store the embedding and previous sublayer outputs as $$v_i$$, and give sublayer $$\ell$$ a learned query vector $$p_\ell$$. At a fixed token position, the layer takes a softmax-weighted sum across **depth**.

$$
a_{\ell i}=\operatorname{softmax}_{i<\ell}
   \big(p_\ell^\top\operatorname{RMSNorm}(v_i)\big),\qquad
x_\ell=\sum_{i<\ell}a_{\ell i}v_i .
$$

A layer can pull more strongly on a particular earlier feature. Full attention residuals retain every previous output. Block Attention Residuals keep one sum per completed block and the running sum in the current block. That reduces the number of depth entries that must be retained and passed between stages.

<figure markdown="1" style="margin-inline: 0">

![Standard, full-attention, and block-attention residual paths through depth.]({{ '/assets/llm-stroll/attnres.png' | relative_url }}){: width="740" height="409" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Standard, full-attention, and block-attention residual paths through depth.</small></figcaption>

</figure>

## Alternatives to token attention

Softmax attention keeps a separate entry for each past position. One way to save work is to summarize the prefix in a recurrent state. Another is to retain addressable history and restrict which positions a query reads. These choices have different costs and different ways of retrieving old details.

### Fixed-size recurrent state

#### Linear attention as a running sum

Suppose we replace the exponential similarity with a kernel that factors as $$\phi(q)^\top\phi(k)$$, using features with a positive normalizer. The query can now act on a sum built before it arrives.

$$
\begin{aligned}
S_t&=S_{t-1}+\phi(k_t)v_t^\top,\qquad z_t=z_{t-1}+\phi(k_t),\\
o_t&=\frac{S_t^\top\phi(q_t)}{z_t^\top\phi(q_t)}.
\end{aligned}
$$

The numerator expands to the sum of query-key weights times values. Each head stores a fixed-size matrix and a normalizer. With feature widths fixed, work grows linearly in sequence length. As more pairs share the same state, their contributions can interfere.

#### Mamba-2 and gated memory

Think of the state as a store that can forget before it writes. A simplified Mamba-2 state update per head has a scalar decay. A key-side gated linear attention (GLA) update can give each state row its own decay.

$$
S_t=A_t S_{t-1}+k_tv_t^\top,\qquad o_t=S_t^\top q_t,\qquad
A_t=\alpha_t I\ \text{or}\ \operatorname{diag}(\alpha_t).
$$

In the Mamba-2 view, the input and readout projections play the roles of $$k_t$$ and $$q_t$$, with the discretization scale folded into the write. Unrolling the scalar case gives every earlier write a product of the intervening decays. Within a chunk this structure can be computed with matrix multiplications, while chunks pass state to one another.

<figure markdown="1" style="margin-inline: 0">

![Mamba and Mamba-2 blocks. The Mamba-2 layout produces its state-space inputs in parallel.]({{ '/assets/llm-stroll/mamba2.png' | relative_url }}){: width="655" height="333" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Mamba and Mamba-2 blocks. The Mamba-2 layout produces its state-space inputs in parallel.</small></figcaption>

</figure>

#### DeltaNet and Gated DeltaNet

Adding a new value at an old key leaves the old value mixed in. DeltaNet first asks what the state currently predicts for that key. It writes back the error, which is one gradient step on the squared reconstruction loss.

$$
\begin{aligned}
L_t(S)&=\tfrac12\lVert S^\top k_t-v_t\rVert_2^2,\\
S_t&=S_{t-1}+\beta_t k_t\big(v_t-S_{t-1}^\top k_t\big)^\top .
\end{aligned}
$$

For $$\lVert k_t\rVert_2=1$$, the readout along $$k_t$$ becomes $$(1-\beta_t)S_{t-1}^\top k_t+\beta_t v_t$$. Gated DeltaNet (GDN) does the repair after decaying the old state by a scalar $$\alpha_t$$.

$$
S_t=(I-\beta_t k_tk_t^\top)\,\alpha_t S_{t-1}+\beta_t k_tv_t^\top .
$$

The decay controls how long memories last. The rank-one correction controls how a write changes the association for this key.

<figure markdown="1" style="margin-inline: 0">

![Two hybrid stacks, with the Gated DeltaNet block on the right.]({{ '/assets/llm-stroll/gdn.png' | relative_url }}){: width="665" height="351" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Two hybrid stacks, with the Gated DeltaNet block on the right.</small></figcaption>

</figure>

#### Kimi Delta Attention and hybrid stacks

Kimi Delta Attention (KDA) gives each key channel its own decay. The diagonal gate acts first, then the same rank-one correction writes the new value.

$$
S_t=(I-\beta_t k_tk_t^\top)\,
       \operatorname{diag}(\alpha_t)S_{t-1}+\beta_t k_tv_t^\top .
$$

Kimi Linear interleaves KDA with a smaller number of multi-head latent attention (MLA) layers. In this hybrid, KDA carries positional information and the MLA layers omit RoPE. The recurrent layers carry a fixed-size state per head. The MLA layers keep position-addressable history in a compact cache, so the hybrid still has a cache term that grows with context length.

<figure markdown="1" style="margin-inline: 0">

![Kimi Linear interleaves MLA with more frequent KDA blocks and uses expert MLPs in both.]({{ '/assets/llm-stroll/kimi-linear.png' | relative_url }}){: width="600" height="630" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>Kimi Linear interleaves MLA with more frequent KDA blocks and uses expert MLPs in both.</small></figcaption>

</figure>

### Sparse softmax in DeepSeek

DeepSeek-V3.2 uses a lightweight indexer to rank previous positions and runs the expensive MLA attention on its top candidates. If the indexer returns a set $$\mathcal I_t$$, the main head computes

$$
o_t=\sum_{j\in\mathcal I_t}
       \frac{\exp(q_t^\top k_j/\sqrt{d_h})}
       {\sum_{s\in\mathcal I_t}\exp(q_t^\top k_s/\sqrt{d_h})}\,v_j .
$$

The indexer first learns to match the full attention's distribution over keys, then the model trains with sparse reads. With a fixed selection size, the main attention reads only that many entries per query. An indexer that scans all earlier positions still has quadratic sequence cost in prefill. DeepSeek-V4 also compresses groups of positions before retrieval. The cache consequences are in [Part II]({{ '/writing/llm-infrastructure/#deepseeks-long-context-cache' | relative_url }}).

<figure markdown="1" style="margin-inline: 0">

![An indexer selects cached positions for the main MLA computation in DeepSeek-V3.2.]({{ '/assets/llm-stroll/dsa.png' | relative_url }}){: width="730" height="403" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>An indexer selects cached positions for the main MLA computation in DeepSeek-V3.2.</small></figcaption>

</figure>

## Reusing layers across depth

### Looped Transformers

A stacked Transformer can give each block its own weights. A **looped Transformer**, also called a recurrent-depth Transformer, runs a shared block or group of blocks several times over a hidden representation. One design uses a short input stack, a recurrent core, and an output stack.

$$
e=P_\psi(x),\qquad
h^{(r+1)}=F_\theta(h^{(r)},e),\qquad
\text{logits}=C_\omega\big(h^{(R)}\big).
$$

A loop refines the hidden state at the same token positions. In a model trained for variable recurrence, increasing $$R$$ spends more compute without adding weights. Each pass still runs the block's attention and MLP, and the work and intermediate state must be accounted for.

<figure markdown="1" style="margin-inline: 0">

![A shared recurrent core receives the input representation on each pass.]({{ '/assets/llm-stroll/looped.png' | relative_url }}){: width="725" height="183" style="height:auto" loading="lazy"}

<figcaption style="text-align: center"><small>A shared recurrent core receives the input representation on each pass.</small></figcaption>

</figure>

The same diagram gives us the next question. Which of its operations can run in parallel, and what must move between devices? Continue in [Part II — Infrastructure]({{ '/writing/llm-infrastructure/' | relative_url }}).
