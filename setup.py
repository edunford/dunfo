from distutils.core import setup
setup(
  name = 'edpy',
  packages = ['edpy'],
  version = '0.0.1',
  license='MIT',
  description = 'A python module built and curated with only one person in mind. Module contains  miscellaneous methods that I use on the reg',
  author = 'Eric Dunford',
  author_email = 'ethomasdunford@gmail.com',
  url = 'https://github.com/edunford/edpy',
  download_url = '',
  keywords = ['tidy', 'misc','wrangling','data','pandas'],
  install_requires=[
          'pandas',
          'sklearn',
      ],
  classifiers=[
    # Status of the project
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Data Science :: Social Science',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Python versions
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
