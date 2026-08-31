---
layout: default
title: Writing
permalink: /writing/
description: "Mathematical and technical writing."
---

<div class="page-title">
  <p class="eyebrow">Writing</p>
  <h1>Articles & Notes</h1>
  <p>Organized by subject. No chronological feed.</p>
</div>

{% assign ordered = site.articles | sort: "order" %}
{% for category in site.article_categories %}
<section class="writing-group">
  <h2>{{ category }}</h2>
  <div class="article-list">
    {% assign found = false %}
    {% for article in ordered %}
      {% if article.category == category %}
        {% assign found = true %}
        <a class="article-row" href="{{ article.url | relative_url }}">
          <div>
            <h3>{{ article.title }}</h3>
            {% if article.description %}<p>{{ article.description }}</p>{% endif %}
          </div>
          {% if article.level %}<span class="category-pill">{{ article.level }}</span>{% endif %}
        </a>
      {% endif %}
    {% endfor %}
    {% unless found %}<p class="empty-note">No article here yet.</p>{% endunless %}
  </div>
</section>
{% endfor %}
