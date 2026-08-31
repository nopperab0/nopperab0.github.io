# 怎么用这个博客（GitHub 网页版即可）

这个站点专门按“文章集”设计，不使用 Jekyll 的 `_posts`，所以文章不需要日期，也不会显示发布日期。

## 0. 第一次：创建网站仓库

1. 登录 GitHub。
2. 新建仓库，名字必须是：`nopperab0.github.io`
3. 选择 **Public**。
4. 创建仓库。
5. 解压本 ZIP，把解压后目录里的**所有文件和文件夹**上传到仓库根目录并 Commit。
6. 进入仓库 **Settings → Pages**。
7. 在 Build and deployment 中选择 **Deploy from a branch**，Branch 选 `main`，目录选 `/(root)`，保存。
8. 等 1–3 分钟，访问 `https://nopperab0.github.io`。

> GitHub 的界面可能略有变化。如果 Pages 已经自动部署，就不需要重复设置。

## 1. 发一篇新文章

在仓库中进入 `_articles` 文件夹：

1. `Add file` → `Create new file`
2. 文件名例如：`green-functions.md`
3. 粘贴下面的模板
4. `Commit changes`
5. 等 GitHub Pages 重新构建，文章会自动出现在 Writing 页面。

模板：

```markdown
---
title: "Green Functions on Infinite Lattices"
subtitle: "Optional subtitle"
category: "Mathematical Physics"
level: "Expository"
order: 4
description: "One-sentence description shown on the article list."
---

* TOC
{:toc}

## 1. Introduction

正文。

行内公式：$E=mc^2$。

独立公式：

$$
\Delta G(x)=\delta_{x,0}.
$$

<div class="theorem">
<strong>Theorem.</strong> 在这里写定理。
</div>

<div class="proof">
<strong>Proof.</strong> 在这里写证明。 ∎
</div>
```

### 注意

- **不要写 `date:`**。
- 文件名也不需要 `2026-08-31-xxx.md` 这种日期格式。
- `order:` 决定文章列表顺序。数字越小越靠前。
- `category:` 当前建议用：`Mathematical Physics`、`Machine Learning`、`Notes`。

## 2. 改文章

打开 `_articles/某篇文章.md` → 点铅笔图标 Edit → 修改 → Commit changes。

## 3. 删除文章

打开文章源码文件 → 右上角菜单 / 删除按钮 → Commit changes。

## 4. 插入图片

先把图片上传到：

`assets/images/`

例如：

`assets/images/lattice.png`

文章中写：

```markdown
![Infinite lattice](/assets/images/lattice.png)
```

## 5. 放 PDF

把 PDF 上传到：

`assets/papers/`

文章中写：

```markdown
[Download the PDF](/assets/papers/my-paper.pdf)
```

## 6. 改名字、简介、GitHub

打开 `_config.yml`：

```yaml
title: "Noppera-bō"
tagline: "Physics · Mathematics · Machine Learning"
description: "..."
author: "Noppera-bō"
github_username: "nopperab0"
```

改完 Commit 即可。

## 7. 改 About

编辑：

`about/index.md`

不用碰 HTML。

## 8. 改主页介绍

编辑：

`index.md`

主页的文章列表会自动读取 `_articles/`，不需要手动加链接。

## 9. 数学公式

本站已加载 MathJax。

行内：

```text
$E=mc^2$
```

独立公式：

```text
$$
R_{\mathrm{eff}}(a,b)=(e_a-e_b)^T L^\dagger(e_a-e_b)
$$
```

你平时写 Markdown 时直接这样写即可。

## 10. 完全不显示文章时间

本站：

- 不使用 `_posts`
- 页面没有 `<time>` 标签
- Front Matter 没有 `date`
- Open Graph 没有 `article:published_time`
- sitemap 没有 `lastmod`
- Writing 页面不按日期排序，只按 `order` 排序

但需要区分一件事：**GitHub 仓库本身的 Commit History 会显示每次提交的时间**。普通访客在你的博客网页上看不到发布日期；如果有人专门进入公开 GitHub 仓库查看提交历史，仍然可以看到文件大致何时被提交。

## 11. 自定义域名（以后再做）

等 `nopperab0.github.io` 正常运行后，再买域名并在 GitHub Pages 的 Custom domain 中绑定即可。网站源代码不需要重写。
