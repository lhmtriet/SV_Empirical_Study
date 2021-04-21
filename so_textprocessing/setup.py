from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(name='so_processing',
    version='0.1',
    description='SO text processing',
    url='https://github.cs.adelaide.edu.au/a1720858/so_textprocessing',
    long_description=readme,
    author='David Hin',
    author_email='a1720858@student.adelaide.edu.au',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['nltk','pyahocorasick','scikit-learn','importlib_resources','pandas','regex','progressbar2'],
    zip_safe=False,
    include_package_data=True
)