setup:
	pip install -r requirements.txt

BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

ifneq (,$(findstring release-,$(BRANCH)))
VERSION := $(subst release-,,$(BRANCH))
else ifneq (,$(findstring hotfix-,$(BRANCH)))
VERSION := $(subst hotfix-,,$(BRANCH))
endif

bump:
	sed -i '' 's/__version__ = .*/__version__ = '\'$(VERSION)\''/' **/*.py
	sed -i '' 's/__version__ = .*/__version__ = '\'$(VERSION)\''\\n",/' spotify-audio-features.ipynb
	autopep8 -i -a -a **/*.py
	pdoc -o ./docs --docformat numpy spotify_audio_features
	pip freeze > requirements.txt
	git add .
	git commit -m "Bump version number to $(VERSION)"
	git checkout master
	git merge $(BRANCH)
	git tag $(VERSION)
	git checkout develop
	git merge $(BRANCH)
	git branch -d $(BRANCH)