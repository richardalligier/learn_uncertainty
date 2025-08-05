.SECONDARY:

include CONFIG


#FOLDER_JSON = ./data
# FOLDER_JSON = ./jsonsmall
#FOLDER_JSON = ./july_test_2

THRESH_ALT= 800
NCLOSEST = 10
THRESH_ALT_CLOSEST= 1800
NSITUATION_MAX = 2

KEEP_DURATION_LONGER_THAN = 120

FOLDER_ALLJSON = $(FOLDER)/json

FOLDER_JSON = $(FOLDER)/json_selected_$(KEEP_DURATION_LONGER_THAN)

NAME_SIT_RAW = situations_$(THRESH_ALT)_$(KEEP_DURATION_LONGER_THAN)

FOLDER_SIT_RAW = $(FOLDER)/$(NAME_SIT_RAW)

FOLDER_SIT = $(FOLDER)/$(NAME_SIT_RAW)_$(NCLOSEST)_$(THRESH_ALT_CLOSEST)

# FILES = $(shell cd $(FOLDER_JSON);  ls *.json)
FILES = $(shell cd $(FOLDER_JSON);  find -type f,l | grep .json$ | cut -c 3-)
#FILES = $(shell cd $(FOLDER_SIT_RAW);  find -type f | grep .situation$ | cut -c 3-)

FILES_SIT = $(foreach f, $(FILES), $(FOLDER_SIT)/$(f:.json=.situation))

#FILES_SIT = $(foreach f, $(FILES), $(FOLDER_SIT)/$(f:.json=.situation))


TARGET_FILE = $(FOLDER)/all_$(THRESH_ALT)_$(NCLOSEST)_$(THRESH_ALT_CLOSEST)_$(NSITUATION_MAX).dsituation


all: $(TARGET_FILE)

figuresdeviated:
	python3 figures.py -situation /disk2/jsonKimdebugBeacons/situations_800_120/2201/34774398_1645197342_1645197662.situation -json /disk2/jsonKimdebugBeacons/json/2201/34774398_1645197342_1645197662.json


figuresothers:
	python3 figures_others_vertical.py -situation /disk2/jsonKimdebugBeacons/situations_800_120/2201/34774398_1645197342_1645197662.situation -json /disk2/jsonKimdebugBeacons/json/2201/34774398_1645197342_1645197662.json
	python3 figures_others.py -situation /disk2/jsonKimdebugBeacons/situations_800_120/2201/34774398_1645197342_1645197662.situation -json /disk2/jsonKimdebugBeacons/json/2201/34774398_1645197342_1645197662.json
	# python3 figures_others.py -situation /disk2/jsonKimdebugBeacons/situations_800_120/2202/35436110_1647850652_1647850928.situation -json /disk2/jsonKimdebugBeacons/json/2202/35436110_1647850652_1647850928.json

analyse:
	python3 analyse_uncertainty.py -statscsv ./stats.csv

select:
	python3 select_situations.py -jsonfolderin $(FOLDER_ALLJSON) -jsonfolderout $(FOLDER_JSON) -keep_duration_longer_than $(KEEP_DURATION_LONGER_THAN)

#-nclosest $(NCLOSEST) -thresh_z $(THRESH_ALT_CLOSEST)

$(FOLDER_SIT)/%.situation: $(FOLDER_SIT_RAW)/%.situation
	mkdir -p $(dir $@)
	python3	reduce.py -situationin $< -situationout $@ -nclosest $(NCLOSEST) -thresh_z $(THRESH_ALT_CLOSEST)

$(FOLDER_SIT_RAW)/%.situation: $(FOLDER_JSON)/%.json
	mkdir -p $(dir $@)
	python3	fit_traj.py -json $< -situation $@ -cutalt $(THRESH_ALT)

$(TARGET_FILE): $(FILES_SIT)
	python3 merge.py -situationsfolder $(FOLDER_SIT) -dsituation $@  -nsituation_max $(NSITUATION_MAX)
