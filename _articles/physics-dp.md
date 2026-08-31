---
title: "Dynamic Programming from a Physicist's Perspective"
subtitle: "State, sufficient information, and recursive structure"
category: "Machine Learning"
level: "Notes"
order: 2
description: "A placeholder article illustrating a second category. Replace it with your own text."
---

> **Sample article.** Delete or overwrite this file when you publish your own note.

## State as sufficient information

A useful way to read a dynamic program is to ask what information about the past must be retained so that the future no longer depends on the full history.

If the state is $s$, a Bellman-style recursion has the generic form

$$
V(s)=\min_a\left\{c(s,a)+V(T(s,a))\right\}.
$$

The difficult part is usually not the recursion itself, but identifying a state that is both sufficient and small.

<div class="remark">
<strong>Remark.</strong> The article system does not require a date field. Ordering is controlled by the integer <code>order</code> in the file header.
</div>
