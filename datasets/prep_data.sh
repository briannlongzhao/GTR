GTR_DIR=~/GTR

MOT17 () {
	DATASETS_DIR=$1/datasets
	cd $DATASETS_DIR || exit
	if ! [ -d "mot/" ]
	then
		mkdir mot
        echo "Downloading MOT17 dataset..."
		wget -q https://motchallenge.net/data/MOT17.zip
        echo "Unzipping MOT17 dataset..."
		unzip -q MOT17.zip -d mot
        cd mot/MOT17 || exit
        ln -s train trainval
        cd $1 || exit
        python tools/convert_mot2coco.py val
        python tools/convert_mot2coco.py test
    fi
}

CrowdHuman () {
	DATASETS_DIR=$1/datasets
    cd $DATASETS_DIR || exit
	if ! [ -d "crowdhuman/" ]
	then
        echo "Downloading CrowdHuman dataset..."
		gdown -q --folder 1-N59uI5plTXEepaIasH3kpFW6SPKSVkQ
		cd crowdhuman || exit
        if ! [ -f "CrowdHuman_train01.zip" ]
        then
            cp $GTR_DIR/datasets/crowdhuman/CrowdHuman_train01.zip $DATASETS_DIR/crowdhuman/
        fi
        if ! [ -f "CrowdHuman_train02.zip" ]
        then
            cp $GTR_DIR/datasets/crowdhuman/CrowdHuman_train02.zip $DATASETS_DIR/crowdhuman/
        fi
        if ! [ -f "CrowdHuman_train03.zip" ]
        then
            cp $GTR_DIR/datasets/crowdhuman/CrowdHuman_train03.zip $DATASETS_DIR/crowdhuman/
        fi
        if ! [ -f "CrowdHuman_val.zip" ]
        then
            cp $GTR_DIR/datasets/crowdhuman/CrowdHuman_val.zip $DATASETS_DIR/crowdhuman/
        fi
        echo "Unzipping Crowdhuman dataset..."
		unzip -q CrowdHuman_train\* -d CrowdHuman_train
		unzip -q CrowdHuman_val\* -d CrowdHuman_val
		cd $1 || exit
		python tools/convert_crowdhuman_amodal.py
	fi
}

BDD100K () {
	DATASETS_DIR=$1/datasets
	cd $DATASETS_DIR || exit
	if ! [ -d "bdd100k/" ]
	then
		mkdir bdd100k
        echo "Downloading BDD100K dataset..."
		gdown -q 1rPJj_OJ-QOMeDyYYBE00V1m-Z0DMfaP5
		gdown -q 1dYeF9b2MSHZiTLtBgszHrzHj3ZjnaTvE
        echo "Unzipping BDD100K dataset..."
		unzip -q bdd100k_images_100k.zip bdd100k_labels_release.zip

        cd bdd100k/MOT17 || exit
        ln -s train trainval
        cd $1 || exit

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
    module load gcc
    module load unzip
elif [[ $HOSTNAME =~ "turing" || $HOSTNAME =~ "vista" ]]
then
    :
else
    echo "Unknown host: $HOSTNAME"
fi

# Download to data directory specified by $TMPDIR if set
if [ -v TMPDIR ]
then 
	echo "Downloading data to TMPDIR=$TMPDIR"
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
else
    echo "TMPDIR not set"
fi
