"""
Tools for making nice, boss-friendly summaries of the statistics.
"""
from jinja2 import Template
import time

def render_info_page(results):
    study_name = results.study.name
    try:
        num_documents = results.stats['num_documents']
        num_concepts = results.stats['num_concepts']
        consistency = results.stats['consistency']
        centrality = results.stats['centrality']
        correlation = results.stats['correlation']
    except KeyError:
        return ""
    key_concepts = {}
    for key, values in results.stats['key_concepts'].items():
        kstr = [u"%s (%1.1f%%)" % (concept, float(value*100))
                for concept, value in values]
        key_concepts[key] = kstr
    core = results.stats['core']
    canonical = centrality.keys()

    timestamp = time.strftime("%I:%M %p, %B %d, %Y", tuple(results.stats['timestamp']))
    return template.render(locals())

def default_info_page(study):
    study_name = study.name
    return default_template.render(locals())

template = Template(u"""
<html>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/></head>
<body>
<h2>Results for {{study_name}}</h2>
<p>Analyzed at {{timestamp}}</p>

<h3>Overall data</h3>
<ul>
  <li>Analyzed {{num_concepts}} concepts from {{num_documents}} documents.</li>
  {% if consistency %}
    <li>Consistency = {{consistency|round(2)}}</li>
    <li>Core concepts: {{core|join(', ')}}</li>
  {% endif %}
</ul>

{% if canonical %}
  <h3>Canonical documents</h3>
  <dl>
    {% for docname in canonical %}
      <dt><b>{{docname}}</b></dt>
      <dd>Centrality: {{centrality[docname]|round(2)}}</dd>
      <dd>Correlation: {{correlation[docname]|round(2)}}</dd>
      <dd>Interesting concepts: {{key_concepts[docname]|join(', ')}}</dd>
    {% endfor %}
  </dl>
  <p>High centrality (&gt; 2.0) means that the document is typical; its
  semantics agree with the semantics of other documents. Low centrality means
  the document is atypical.</p>
{% endif %}
</body></html>
""")

default_template = Template(
"""
<html><body>
<h2>New study: {{study_name}}</h2>
<p>This study has not been analyzed yet. Click the <b>Analyze</b> button
on the toolbar to do so.</p>
</body></html>
""")

