from distutils.core import setup
setup(
  name = 'komori',
  packages = ['komori'], # this must be the same as the name above
  version = '0.19',
  description = 'komori: bat call recording visualisation package',
  author = 'Domagoj K. Hackenberger',
  author_email = 'domagojhack@gmail.com',
  url = 'https://github.com/BioQuantHr/komori', # use the URL to the github repo
  download_url = 'https://github.com/BioQuantHr/komori/archive/0.19.tar.gz', # I'll explain this in a second
  keywords = ['STFT', 'spectrogram', 'bat', 'calls', 'spectrograms', 'bioacoustics', 'ultrasound'], # arbitrary keywords
  classifiers = [],
  install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
      ],
)
