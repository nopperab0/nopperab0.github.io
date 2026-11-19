# Editing the blog

The homepage lists papers, project repositories, and notes. Notes are published from Jekyll's `_articles» collection.

## GitHub Pages setup

1. Put the site files in the root of the public `nopperab0.github.io» repository.
2. Open **Settings → Pages** and choose **Deploy from a branch**.
3. Select `main» and `/(root)», then save. [Open the blog](https://nopperab0.github.io/)

## Papers and projects

Edit the `papers» and `projects» lists in `index.md». Each entry has a title and an absolute URL. Papers also have a source label.

## Notes

Add a Markdown file such as `_articles/green-functions.md». The note appears under Notes on the homepage after the Pages build. The `order» field controls its position in the list.

~~~markdown
---
title: "Green Functions on Infinite Lattices"
subtitle: "Optional subtitle"
category: "Mathematical Physics"
order: 4
description: "A short description for the homepage."
---

* TOC
{:toc}

## Introduction

Inline math uses $$E=mc^2$$.

$$
\Delta G(x)=\delta_{x,0}.
$$

<div class="theorem">
<strong>Theorem.</strong> State the result here.
</div>

<div class="proof">
<strong>Proof.</strong> Write the proof here.
</div>
~~~

Edit or delete the Markdown file in `_articles» and commit the change to update or remove a note. The site omits publication dates. GitHub commit history records when files changed.

## Images and PDFs

Put images in `assets/images/» and PDFs in `assets/papers/». Link to the file from the note.

~~~markdown
![Infinite lattice](/assets/images/lattice.png)

[Download the PDF](/assets/papers/my-paper.pdf)
~~~

## Site settings

Edit `_config.yml» to change the site title, tagline, author, or description. Edit the headline area in `index.md» to change the homepage introduction. Set a custom domain under **Settings → Pages** when one is available.
