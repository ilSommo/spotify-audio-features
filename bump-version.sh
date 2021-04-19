#!/bin/bash

sed -i '' 's/__version__ = .*/__version__ = '$1'/' functions.py
sed -i '' 's/__version__ = .*/__version__ = '$1'/' spotify-audio-features.ipynb

git commit -a -m "Bump version number to "$1""