---
layout: default
title:
permalink: /
---

<section class="hero">
  <p class="eyebrow">Personal mathematical writing</p>
  <h1>{{ site.title }}</h1>
  <p class="hero-tagline">{{ site.tagline }}</p>
  <p class="hero-copy">Expository notes, derivations, and research-adjacent writing. The site is organized by subject rather than publication date.</p>
  <div class="hero-actions">
    <a class="button primary" href="{{ '/writing/' | relative_url }}">Read writing</a>
    <a class="button" href="{{ '/about/' | relative_url }}">About</a>
  </div>
</section>

<section class="section-block">
  <div class="section-heading">
    <h2>Selected writing</h2>
    <a href="{{ '/writing/' | relative_url }}">All articles →</a>
  </div>

  <div class="article-list">
    {% assign ordered = site.articles | sort: "order" %}
    {% for article in ordered limit: 5 %}
      <a class="article-row" href="{{ article.url | relative_url }}">
        <div>
          <h3>{{ article.title }}</h3>
          {% if article.description %}<p>{{ article.description }}</p>{% endif %}
        </div>
        <span class="category-pill">{{ article.category }}</span>
      </a>
    {% endfor %}
  </div>
</section>

<section class="section-block compact-note">
  <h2>What lives here</h2>
  <p>Long-form mathematical notes, proofs, physical intuition, and technical essays. Articles support displayed equations, theorem boxes, figures, code blocks, and downloadable PDFs.</p>
</section>
