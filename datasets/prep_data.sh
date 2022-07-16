GTR_DIR=~/GTR

pip install gdown
pip install opencv-python

MOT17 () {
	DATASETS_DIR=$1/datasets
	cd $DATASETS_DIR || exit
	if ! [ -d "mot/" ]
	then
		mkdir mot
		wget https://motchallenge.net/data/MOT17.zip
		unzip -q MOT17.zip -d mot
	    cd mot/MOT17 || exit
	    ln -s train trainval
	    cd $1 || exit
	    python tools/convert_mot2coco.py
	    rm $DATASETS_DIR/*.zip
    fi
}

CrowdHuman () {
	DATASETS_DIR=$1/datasets
    cd $DATASETS_DIR || exit
	if ! [ -d "crowdhuman/" ]
	then 
		gdown --folder 1-N59uI5plTXEepaIasH3kpFW6SPKSVkQ
		cd crowdhuman || exit
		unzip -q CrowdHuman_train\* -d CrowdHuman_train
		unzip -q CrowdHuman_val\* -d CrowdHuman_val
		cd $1 || exit
		python tools/convert_crowdhuman_amodal.py
		rm $DATASETS_DIR/crowdhuman/*.zip
	fi
}

# Download to home directory
#MOT17 $GTR_DIR
#CrowdHuman $GTR_DIR

# Set TMPDIR if on iLab or Discovery
if [[ $HOSTNAME =~ iGpu || $HOSTNAME =~ iLab ]]
then
	export TMPDIR=/lab/tmpig8e/u/brian-data
elif [[ $HOSTNAME =~ "discovery" || $HOSTNAME =~ "hpc" || $HOSTNAME =~ [a-z][0-9][0-9]-[0-9][0-9] ]]
then
  export TMPDIR=/scratch1/briannlz
fi

# Download to data directory specified by $TMPDIR if set
if [ -v TMPDIR ]
then 
	echo "TMPDIR is set to '$TMPDIR'"
	GTR_DIR_TMP=$TMPDIR/GTR

	# Copy datasets metadata to $TMPDIR
	if ! [ -d $GTR_DIR_TMP/datasets ]
	then 
		mkdir -p $GTR_DIR_TMP/datasets
		cp -r $GTR_DIR/datasets/metadata $GTR_DIR_TMP/datasets/metadata
	fi

	# Copy tools to $TMPDIR
	if ! [ -d $GTR_DIR_TMP/tools ]
	then
		cp -r $GTR_DIR/tools $GTR_DIR_TMP/tools
	fi

	# Copy models to $TMPDIR
	if ! [ -d $GTR_DIR_TMP/models ]
	then
		cp -r $GTR_DIR/models $GTR_DIR_TMP/models
	fi

	MOT17 $GTR_DIR_TMP
	CrowdHuman $GTR_DIR_TMP
fi
