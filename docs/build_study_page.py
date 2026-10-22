"""Build docs/index.html (the interactive study page) from docs/study_data.json.

Usage:
    python docs/build_study_page.py                      # rebuild the page from study_data.json
    python docs/build_study_page.py --from-extras DIR    # first refresh study_data.json from extras JSON files

The page is a single self-contained HTML file: open it straight from disk or via GitHub Pages.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'study_data.json')
OUT = os.path.join(HERE, 'index.html')
REPO_URL = 'https://github.com/ogjr80/mlproject/blob/main/'

ORDER = ['data_preprocessing_tools', 'data_preprocessing_template', 'simple_linear_regression',
         'multiple_linear_regression', 'polynomial_regression', 'support_vector_regression',
         'decision_tree_regression', 'random_forest_regression', 'logistic_regression',
         'k_nearest_neighbors', 'support_vector_machine', 'kernel_svm', 'naive_bayes',
         'decision_tree_classification', 'random_forest_classification', 'k_means_clustering',
         'hierarchical_clustering', 'apriori', 'eclat', 'upper_confidence_bound', 'thompson_sampling',
         'natural_language_processing', 'artificial_neural_network', 'convolutional_neural_network',
         'principal_component_analysis', 'linear_discriminant_analysis', 'kernel_pca',
         'k_fold_cross_validation', 'grid_search', 'xg_boost', 'graph_transformer']


def from_extras(folder):
    items = {}
    for f in glob.glob(os.path.join(folder, '*.json')):
        ex = json.load(open(f, encoding='utf-8'))
        stem = os.path.basename(ex['notebook'])[:-6]
        part, section = ex['notebook'].split('/')[:2]
        nb = json.load(open(os.path.join(HERE, '..', ex['notebook']), encoding='utf-8'))
        title = next(''.join(c['source']).lstrip('# ').strip() for c in nb['cells']
                     if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('# '))
        items[stem] = {
            'id': stem, 'title': title, 'part': part.split(' - ', 1)[-1], 'partNo': part.split(' - ')[0],
            'section': section, 'url': REPO_URL + ex['notebook'].replace(' ', '%20'),
            'summary': ex['summary'], 'whenToUse': ex['when_to_use'],
            'readFirst': ex['before_you_start']['read_first'], 'terms': ex['before_you_start']['new_terms'],
            'mistakes': ex['common_mistakes'], 'tryThis': ex['try_this'], 'quiz': ex['quiz'],
        }
    data = [items[k] for k in ORDER if k in items]
    with open(DATA, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
        fh.write('\n')
    print(f'study_data.json: {len(data)} notebooks')


def build():
    data = json.load(open(DATA, encoding='utf-8'))
    template = open(os.path.join(HERE, 'study_template.html'), encoding='utf-8').read()
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(template.replace('/*__DATA__*/[]', payload))
    print(f'index.html: {len(data)} notebooks, {os.path.getsize(OUT) // 1024} KB')


if __name__ == '__main__':
    if '--from-extras' in sys.argv:
        from_extras(sys.argv[sys.argv.index('--from-extras') + 1])
    build()
