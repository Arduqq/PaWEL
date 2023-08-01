import setuptools
with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()
setuptools.setup(name='PaWEL',
                 description='Webcam processing for physiological response detection',
                 packages=['PaWEL'],
                 install_requires=install_requires)