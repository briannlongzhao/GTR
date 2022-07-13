GTR_DIR=~/GTR

MOT17 () {
	DATASETS_DIR=$1/datasets
	cd $DATASETS_DIR
	if ! [ -d "mot/" ]
	then
		mkdir mot
		wget https://motchallenge.net/data/MOT17.zip
		unzip -q MOT17.zip -d mot
	    cd mot/MOT17
	    ln -s train trainval
	    cd $1
	    python tools/convert_mot2coco.py
	    rm $DATASETS_DIR/*.zip
    fi
}


CrowdHuman () {
	DATASETS_DIR=$1/datasets
    cd $DATASETS_DIR
	if ! [ -d "crowdhuman/" ]
	then 
		gdown --folder 1-N59uI5plTXEepaIasH3kpFW6SPKSVkQ
		cd crowdhuman
		unzip -q CrowdHuman_train\* -d CrowdHuman_train
		unzip -q CrowdHuman_val\* -d CrowdHuman_val
		cd $1
		python tools/convert_crowdhuman_amodal.py
		rm $DATASETS_DIR/crowdhuman/*.zip
	fi
}


MOT17 $GTR_DIR
CrowdHuman $GTR_DIR

if [ -v TMPDIR ]
then 
	echo "TMPDIR is set to '$TMPDIR'"
	GTR_DIR_TMP=$TMPDIR/GTR
	if ! [ -d $GTR_DIR_TMP/datasets ]
	then 
		mkdir -p $GTR_DIR_TMP/datasets
		cp -r $GTR_DIR/datasets/metadata $GTR_DIR_TMP/datasets/metadata
	fi
	if ! [ -d $GTR_DIR_TMP/tools ]
	then
		cp -r $GTR_DIR/tools $GTR_DIR_TMP/tools
	fi
	MOT17 $GTR_DIR_TMP
	CrowdHuman $GTR_DIR_TMP
fi
