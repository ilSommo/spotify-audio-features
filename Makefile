setup:
	pip install -r requirements.txt

BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

ifneq (,$(findstring release-,$(BRANCH)))
VERSION := $(subst release-,,$(BRANCH))
else ifneq (,$(findstring hotfix-,$(BRANCH)))
VERSION := $(subst hotfix-,,$(BRANCH))
endif

bump: $(shell find . -name "*.py")
	sed -i '' 's/__version__ = .*/__version__ = '\'$(VERSION)\''/' $^
	sed -i '' 's/__version__ = .*/__version__ = '\'$(VERSION)\''\\n",/' spotify-audio-features.ipynb
	autopep8 -i -a -a $^
	pdoc -o ./docs --docformat numpy spotify_audio_features
	pip freeze > requirements.txt
	cd report && make all
	git add .
	git commit -m "Bump version number to $(VERSION)"
	git checkout master
	git merge $(BRANCH)
	git tag $(VERSION)
	git checkout develop
	git merge $(BRANCH)
	git branch -d $(BRANCH)
