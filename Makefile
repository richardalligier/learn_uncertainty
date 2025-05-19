#FOLDER_JSON = ./data
# FOLDER_JSON = ./jsonsmall
#FOLDER_JSON = ./july_test_2

THRESH_ALT= 800
FOLDER_JSON = /disk2/jsonKim/json

FOLDER_SIT = /disk2/jsonKim/situations_$(THRESH_ALT)

# FILES = $(shell cd $(FOLDER_JSON);  ls *.json)
FILES = $(shell cd $(FOLDER_JSON);  find -type f | grep .json$ | cut -c 3-)

FILES_SIT = $(foreach f, $(FILES), $(FOLDER_SIT)/$(f:.json=.situation))



all: all_$(THRESH_ALT).dsituation


$(FOLDER_SIT)/%.situation: $(FOLDER_JSON)/%.json
	mkdir -p $(FOLDER_SIT)
	python3	fit_traj.py -json $< -situation $@ -cutalt $(THRESH_ALT)

all_$(THRESH_ALT).dsituation: $(FILES_SIT)
	python3 merge.py -situationsfolder $(FOLDER_SIT) -dsituation $@
