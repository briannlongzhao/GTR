DATASET_DIR=~/GTR/datasets

MOT17 () {
	cd $1
	if [ -d "mot/" ]; then return; fi
	mkdir mot
	wget https://motchallenge.net/data/MOT17.zip
	unzip MOT17.zip -d mot/
	cd $1/mot/MOT17/
	ln -s train trainval
	python $1/../tools/convert_mot2coco.py
	rm ../../*.zip
}


CrowdHuman () {
	cd $1
	if [ -d "crowdhuman/" ]; then return; fi
	gdown --folder 1-N59uI5plTXEepaIasH3kpFW6SPKSVkQ
	cd crowdhuman
	unzip CrowdHuman_train\* -d CrowdHuman_train/
	unzip CrowdHuman_val\* -d CrowdHuman_val/
	python $1/../tools/convert_crowdhuman_amodal.py
	rm *.zip
}

MOT17 $DATASET_DIR
CrowdHuman $DATASET_DIR

if [ -v TMPDIR ]
then 
	echo "TMPDIR is set to '$TMPDIR'"
	cd $TMPDIR
	if ![ -d "GTR/datasets/" ] then mkdir -p GTR/datasets/; fi
	DATASET_DIR_TMP=$TMPDIR/GTR/datasets
	cp -r $DATASET_DIR/metadata/ $DATASET_DIR_TMP/metadata
	cp -r $DATASET_DIR/../tools/ $DATASET_DIR_TMP/../tools/
	MOT17 $DATASET_DIR_TMP
	CrowdHuman $DATASET_DIR_TMP
fi
