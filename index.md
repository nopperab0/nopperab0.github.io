---
layout: default
title:
permalink: /
papers:
  - title: "Convergent Stochastic Training of Multi-Headed Attention and Understanding LoRA"
    url: "https://arxiv.org/abs/2605.07959"
    source: "arXiv"
  - title: "Bias-Reduced Data-Parallel Natural Policy Gradient via Poissonized Determinantal Averaging"
    url: "https://openreview.net/forum?id=AASBoc1f7o"
    source: "OpenReview"
  - title: "Green BOA: Determining the environmental break-even point for ML-based data compression"
    url: "https://arxiv.org/abs/2608.19994"
    source: "arXiv"
projects:
  - title: "BOA"
    url: "https://github.com/nopperab0/boa-constrictor"
  - title: "FM4CAL"
    url: "https://github.com/nopperab0/FM4CAL"
  - title: "NanoSpec"
    url: "https://github.com/nopperab0/NanoSpec"
  - title: "minWM"
    url: "https://github.com/nopperab0/minWM"
  - title: "Perpetual Alpha"
    url: "https://github.com/BouillieAnonymous/Perpetual-Alpha-2026"
  - title: "Monte Carlo Simulation of Thermal Neutron Transport"
    url: "https://github.com/nopperab0/Model-of-Stochastic-Transport-of-Thermal-Neutrons-using-Monte-Carlo"
---

<section class="hero">
  <h1>{{ site.title }}</h1>
  <p class="hero-tagline">{{ site.tagline }}</p>
</section>

<section class="section-block" id="papers" aria-labelledby="papers-title">
  <h2 id="papers-title">Papers</h2>
  <div class="article-list">
    {% for paper in page.papers %}
      <a class="article-row" href="{{ paper.url | escape }}">
        <h3>{{ paper.title | escape }}</h3>
        <span class="category-pill">{{ paper.source }}</span>
      </a>
    {% endfor %}
  </div>
</section>

<section class="section-block" id="projects" aria-labelledby="projects-title">
  <h2 id="projects-title">Projects</h2>
  <div class="article-list">
    {% for project in page.projects %}
      <a class="article-row" href="{{ project.url | escape }}">
        <h3>{{ project.title | escape }}</h3>
      </a>
    {% endfor %}
  </div>
</section>

<section class="section-block" id="notes" aria-labelledby="notes-title">
  <h2 id="notes-title">Notes</h2>
  <div class="article-list">
    {% assign ordered = site.articles | sort: "order" %}
    {% for article in ordered %}
      <a class="article-row" href="{{ article.url | relative_url }}">
        <div>
          <h3>{{ article.title }}</h3>
          {% if article.description %}<p>{{ article.description }}</p>{% endif %}
        </div>
      </a>
    {% endfor %}
  </div>
</section>
