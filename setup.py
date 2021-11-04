# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['sktopic',
 'sktopic.callbacks',
 'sktopic.components',
 'sktopic.distributions',
 'sktopic.metrics',
 'sktopic.models',
 'sktopic.trainers',
 'sktopic.utils']

package_data = \
{'': ['*']}

install_requires = \
['gensim>=4.1.2,<5.0.0',
 'hydra-core>=1.1.1,<2.0.0',
 'hyperspherical-vae @ '
 'git+https://github.com/nicola-decao/s-vae-pytorch.git@master',
 'more-itertools>=8.10.0,<9.0.0',
 'octis>=1.9.0,<2.0.0',
 'pandas>=1.3.4,<2.0.0',
 'seaborn>=0.11.2,<0.12.0',
 'skorch @ git+https://github.com/skorch-dev/skorch.git@release-0.11.0',
 'toolz>=0.11.1,<0.12.0',
 'torch>=1.10.0,<2.0.0',
 'tqdm>=4.62.3,<5.0.0',
 'wandb>=0.12.5,<0.13.0',
 'wordcloud>=1.8.1,<2.0.0']

setup_kwargs = {
    'name': 'sktopic',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'rsimd',
    'author_email': 'rickysimd@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)
