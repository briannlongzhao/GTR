GTR_DIR=~/GTR

pip install gdown
pip install opencv-python

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

# Download to home directory
#MOT17 $GTR_DIR
#CrowdHuman $GTR_DIR

# Set TMPDIR if on ilab
if [[ $HOSTNAME =~ iGpu || $HOSTNAME =~ iLab ]]
then
	export TMPDIR=/lab/tmpig8e/u/brian-data
fi

# Download to data directory specified by $TMPDIR if set
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
	if ! [ -d $GTR_DIR_TMP/models ]
	then
		cp -r $GTR_DIR/models $GTR_DIR_TMP/models
	fi
	MOT17 $GTR_DIR_TMP
	CrowdHuman $GTR_DIR_TMP
fi
