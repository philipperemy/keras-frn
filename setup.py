from setuptools import setup

setup(
    name='frn',
    version='1.0',
    description='Filter Response Normalization Layer: Eliminating '
                'Batch Dependence in the Training of Deep Neural Networks',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['frn'],
    install_requires=[
        'numpy==1.16.2',
        'keras',
        'tensorflow>=2.*'
        'gast==0.2.2'
    ]
)
