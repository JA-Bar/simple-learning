import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(name='simple-learning',
                 version='0.0.1',
                 author='Juan Barajas',
                 description='A basic DL framework for educational purposes.',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/JA-Bar/simple-learning',
                 packages=setuptools.find_packages(),
                 classifiers=[
                    'Programming Language :: Python :: 3'
                 ],
                 python_requires='>=3.6'
                 )

