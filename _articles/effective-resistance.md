---
title: "Effective Resistance on Infinite Lattices"
subtitle: "A sample mathematical article showing the site typography"
category: "Mathematical Physics"
level: "Expository"
order: 1
description: "Discrete Poisson equations, Green functions, and a Fourier viewpoint. Replace this sample with your own article."
---

> **Sample article.** This file exists to show the typography. You can delete it or replace the text with your own writing.

* TOC
{:toc}

## 1. The basic problem

Consider a resistor network on a graph. If a unit current is injected at a vertex $a$ and removed at a vertex $b$, Kirchhoff's law gives a discrete Poisson equation

$$
L V = e_a-e_b,
$$

where $L$ is the graph Laplacian. The voltage difference between the two marked vertices is the effective resistance.

<div class="definition">
<strong>Definition.</strong> The effective resistance between vertices $a$ and $b$ is the voltage drop required to sustain one unit of current from $a$ to $b$.
</div>

For a finite connected graph one obtains

$$
R_{\mathrm{eff}}(a,b)
= (e_a-e_b)^\top L^\dagger(e_a-e_b).
$$

## 2. Fourier representation

On a translation-invariant lattice the discrete Laplacian is diagonal in Fourier space. This turns the Green-function problem into an integral over the Brillouin zone.

<div class="theorem">
<strong>Theorem (sample).</strong> Translation symmetry converts convolution by the lattice Laplacian into multiplication by its Fourier symbol.
</div>

<div class="proof">
<strong>Proof.</strong> Insert a Fourier mode into the discrete Laplacian. Every lattice translation contributes only a phase factor, so the mode is an eigenfunction. ∎
</div>

## 3. A figure

Put an image in `assets/images/` and write

```markdown
![A lattice diagram](/assets/images/your-figure.png)
```

## References

1. Add references as an ordinary numbered list.
2. You can also link to an arXiv page or a local PDF.
