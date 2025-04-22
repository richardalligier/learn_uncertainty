#FOLDER_JSON = ./data
FOLDER_JSON = ./jsonsmall
#FOLDER_JSON = ./july_test_2

FOLDER_SIT = ./situations

FILES = $(shell cd $(FOLDER_JSON);  ls *.json)

FILES_SIT = $(foreach f, $(FILES), $(FOLDER_SIT)/$(f:.json=.situation))



all: all.dsituation


$(FOLDER_SIT)/%.situation: $(FOLDER_JSON)/%.json
	mkdir -p $(FOLDER_SIT)
	python3	fit_traj.py -json $< -situation $@

all.dsituation: $(FILES_SIT)
	python3 merge.py -situationsfolder $(FOLDER_SIT) -dsituation $@
