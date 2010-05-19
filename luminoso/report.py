"""
Tools for making nice, boss-friendly summaries of the statistics.
"""
from jinja2 import Template
import time

def is_a_document(feature):
    return isinstance(feature, basestring) and feature.endswith('.txt')

def render_info_page(study):
    if study.stats is None:
        return render_info_nodocs(study)
    else:
        study_name = study.get_name()
        num_documents = study.num_documents
        
        # FIXME: better way to tell concepts apart from documents
        concepts = [c for c in study.study_concepts if not is_a_document(c)]

        num_concepts = len(concepts)
        canonical = study.canonical_docs
        consistency = study.stats['mean'] / study.stats['stdev']
        congruence = study.stats['congruence']

        sorted_congruence = sorted(concepts,
                                   key=lambda c: congruence.get(c, 0))
        core = sorted_congruence[-10:]
        core.reverse()
        if 'timestamp' in study.stats:
            timestamp = time.strftime("%I:%M %p, %B %d, %Y", tuple(study.stats['timestamp']))
        return template.render(locals())

def render_info_nodocs(study):
    study_name = study.get_name()
    num_concepts = len(study.study_concepts)
    if 'timestamp' in study.stats:
        timestamp = time.strftime("%I:%M %p, %B %d, %Y", tuple(study.stats['timestamp']))
    return nodocs_template.render(locals())

def default_info_page(study):
    study_name = study.get_name()
    return default_template.render(locals())

template = Template("""
<html><body>
<h2>Results for {{study_name}}</h2>
<p>Analyzed at {{timestamp}}</p>

<h3>Overall data</h3>
<ul>
  <li>Extracted {{num_concepts}} concepts from {{num_documents}} documents.</li>
  <li>Consistency = {{consistency|round(2)}}</li>
  <li>Core concepts: {{core|join(', ')}}</li>
</ul>

{% if canonical %}
  <h3>Canonical documents</h3>
  <ul>
    {% for docname in canonical %}
      <li><b>{{docname}}</b>: Congruence = {{congruence[docname]|round(2)}}</li>
    {% endfor %}
  </ul>
  <p>High congruence (&gt; 2.0) means that the document is typical; its
  semantics agree with the semantics of other documents. Low congruence means
  the document is atypical.</p>
{% endif %}
</body></html>
""")

nodocs_template = Template(
"""
<html><body>
<h2>Results for {{study_name}}</h2>
<p>Analyzed at {{timestamp}}</p>

<p>Analyzed {{num_concepts}} concepts, with no additional documents.</p>
""")


default_template = Template(
"""
<html><body>
<h2>New study: {{study_name}}</h2>
<p>This study has not been analyzed yet. Click the <b>Analyze</b> button
on the toolbar to do so.</p>
""")

