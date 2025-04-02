FOLDER_JSON = ./data

FOLDER_SIT = ./situations

FILES = $(shell cd $(FOLDER_JSON);  ls *.json)

FILESI = $(foreach f, $(FILES), $(FOLDER_SIT)/$(f:.json=.situation))

all: $(FILESI)


$(FOLDER_SIT)/%.situation: $(FOLDER_JSON)/%.json
	mkdir -p $(FOLDER_SIT)
	python3	fit_traj.py -json $< -situation $@
