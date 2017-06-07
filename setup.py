from distutils.core import setup
setup(
  name = 'KomoriVis',
  packages = ['KomoriVis'], # this must be the same as the name above
  version = '0.15',
  description = 'KomoriVis: Bat call visualisation package',
  author = 'Domagoj K. Hackenberger',
  author_email = 'domagojhack@gmail.com',
  url = 'https://github.com/BioQuantHr/KomoriVis', # use the URL to the github repo
  download_url = 'https://github.com/BioQuantHr/KomoriVis/archive/0.15.tar.gz', # I'll explain this in a second
  keywords = ['STFT', 'spectrogram', 'bat', 'calls', 'spectrograms', 'bioacoustics', 'ultrasound'], # arbitrary keywords
  classifiers = [],
  install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
      ],
)
