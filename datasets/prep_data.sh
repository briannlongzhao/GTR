GTR_DIR=~/GTR

MOT17 () {
	DATASETS_DIR=$1/datasets
	cd $DATASETS_DIR || exit
	if ! [[ -d mot/ ]]
	then
		mkdir mot/
		if [[ -f $GTR_DIR/datasets/MOT17.zip ]]; then
		    echo "Copying MOT17 dataset..."
		    cp $GTR_DIR/datasets/MOT17.zip ./
		else
		    echo "Downloading MOT17 dataset..."
		    wget -q https://motchallenge.net/data/MOT17.zip
		fi
        echo "Extracting MOT17 dataset..."
		unzip -q MOT17.zip -d mot/
        cd mot/MOT17/ || exit
        ln -s train trainval
        cd $1 || exit
        python tools/convert_mot2coco.py val
        python tools/convert_mot2coco.py test
    fi
}

CrowdHuman () {
	DATASETS_DIR=$1/datasets
    cd $DATASETS_DIR || exit
	if ! [[ -d crowdhuman/ ]]; then
	    CH_COPIED=1
	    mkdir crowdhuman/
	    if [[ -f $GTR_DIR/datasets/crowdhuman/CrowdHuman_train01.zip && CH_COPIED == 1 ]]; then
	        echo "Copying CrowdHuman dataset 1..."
	        cp $GTR_DIR/datasets/crowdhuman/CrowdHuman_train01.zip crowdhuman/ || CH_COPIED=0
	    else
	        CH_COPIED=0
	    fi
	    if [[ -f $GTR_DIR/datasets/crowdhuman/CrowdHuman_train02.zip && CH_COPIED == 1 ]]; then
	        echo "Copying CrowdHuman dataset 2..."
	        cp $GTR_DIR/datasets/crowdhuman/CrowdHuman_train02.zip crowdhuman/ || CH_COPIED=0
	    else
	        CH_COPIED=0
	    fi
	    if [[ -f $GTR_DIR/datasets/crowdhuman/CrowdHuman_train03.zip && CH_COPIED == 1 ]]; then
	        echo "Copying CrowdHuman dataset 3..."
	        cp $GTR_DIR/datasets/crowdhuman/CrowdHuman_train03.zip crowdhuman/ || CH_COPIED=0
	    else
	        CH_COPIED=0
	    fi
	    if [[ -f $GTR_DIR/datasets/crowdhuman/CrowdHuman_val.zip && CH_COPIED == 1 ]]; then
	        echo "Copying CrowdHuman dataset 4..."
	        cp $GTR_DIR/datasets/crowdhuman/CrowdHuman_val.zip crowdhuman/ || CH_COPIED=0
	    else
	        CH_COPIED=0
	    fi
	    if [[ -f $GTR_DIR/datasets/crowdhuman/annotation_train.odgt && CH_COPIED == 1 ]]; then
	        echo "Copying CrowdHuman dataset 5..."
	        cp $GTR_DIR/datasets/crowdhuman/annotation_train.odgt crowdhuman/ || CH_COPIED=0
	    else
	        CH_COPIED=0
	    fi
	    if [[ -f $GTR_DIR/datasets/crowdhuman/annotation_val.odgt && CH_COPIED == 1 ]]; then
	        echo "Copying CrowdHuman dataset 6..."
	        cp $GTR_DIR/datasets/crowdhuman/annotation_val.odgt crowdhuman/ || CH_COPIED=0
	    else
	        CH_COPIED=0
	    fi
	    if [[ $CH_COPIED == 0 ]]; then
	        echo "Downloading CrowdHuman dataset..."
		    gdown -q --folder 1-N59uI5plTXEepaIasH3kpFW6SPKSVkQ
		fi
		cd crowdhuman || exit
		if [[ $(ls -1 | wc -l) != 6 ]]; then
		    echo "Error: Incomplete CrowdHuman dataset"
		    exit
        fi
        echo "Extracting CrowdHuman dataset..."
		unzip -q CrowdHuman_train\* -d CrowdHuman_train
		unzip -q CrowdHuman_val\* -d CrowdHuman_val
		cd $1 || exit
		python tools/convert_crowdhuman_amodal.py
	fi
}

BDD100K () {
	DATASETS_DIR=$1/datasets
	cd $DATASETS_DIR || exit
	if ! [[ -d bdd/ ]]; then
		if [[ $HOSTNAME =~ "turing" || $HOSTNAME =~ "vista" ]]; then
		    if [[ -f /nas/vista-ssd02/users/jmathai/bdd100k_qdtrack_data.tar ]]; then
		        echo "Copying BDD100K dataset..."
		        cp /nas/vista-ssd02/users/jmathai/bdd100k_qdtrack_data.tar ./
		        echo "Extracting BDD100K dataset..."
		        tar -xf bdd100k_qdtrack_data.tar
		        mv data/ bdd/
		        mv bdd/bdd/ bdd/BDD100K/
		        # manage data
		    else
		        echo "BDD100K dataset does not exist, contact Joe"
		        exit
        fi
        if ! [[ -d bdd/ ]]; then
            mkdir bdd/
            cd bdd/
            echo "Downloading BDD100K dataset..."
            gdown -q 1u1DKH7Stk-YNRhhGHHJx_4lUH7sA_e6m
            curl -f -L http://dl.yf.io/bdd100k/mot20/ | grep -oe "images.*"\"$"" > bdd100k.txt
            sed -i 's#"##' bdd100k.txt && sed -i 's#^#http://dl.yf.io/bdd100k/mot20/#' bdd100k.txt
            cat bdd100k.txt | xargs -n 1 -P 3 wget -q
            for file in *; do
                if [[ $file =~ ".md5" ]]; then
                    if ! [[ $(md5sum $(echo $file | sed 's/.md5//') | cut -d' ' -f 1) == $(cat $file) ]]; then
                        echo "Error: MD5 checksum does not match for" $file
                        exit
                    else
                        echo $file "checked"
                    fi
                fi
            done
            rm *.md5 *.txt
            if [[ $(ls -1 | wc -l) != 11 ]]; then
                echo "Error: Incomplete BDD100K dataset"
                exit
            fi
            echo "Extracting BDD100K dataset..."
            unzip -q \*.zip
            mv bdd100k/ BDD100K/
            echo "Converting BDD100K to COCO format..."
		    cd $1/tools || exit
		    python3 -m bdd100k.label.to_coco -m box_track -i $DATASETS_DIR/bdd/BDD100K/labels/box_track_20/train/bdd100k_labels_images_val.json -o $DATASETS_DIR/bdd100k/labels/bdd100k_labels_images_val_coco.json
            python3 -m bdd100k.label.to_coco -m box_track -i $DATASETS_DIR/bdd/BDD100K/labels/bdd100k_labels_images_val.json -o $DATASETS_DIR/bdd100k/labels/bdd100k_labels_images_val_coco.json

        fi


		echo "Converting BDD100K to COCO format..."
		cd $1/tools || exit

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
    echo "Error: Unknown host: $HOSTNAME"
    exit
fi

# Download to data directory specified by $TMPDIR if set
if [[ -v TMPDIR ]]
then 
	echo "Prepaaring data in TMPDIR=$TMPDIR"
	GTR_DIR_TMP=$TMPDIR/GTR

	# Copy datasets metadata to $TMPDIR
	if ! [[ -d $GTR_DIR_TMP/datasets ]]
	then 
		mkdir -p $GTR_DIR_TMP/datasets
		cp -r $GTR_DIR/datasets/metadata $GTR_DIR_TMP/datasets/metadata
	fi

	# Copy tools to $TMPDIR
	if ! [[ -d $GTR_DIR_TMP/tools ]]
	then
		cp -r $GTR_DIR/tools $GTR_DIR_TMP/tools
	fi

	# Copy models to $TMPDIR
	if ! [[ -d $GTR_DIR_TMP/models ]]
	then
		cp -r $GTR_DIR/models $GTR_DIR_TMP/models
	fi

	#MOT17 $GTR_DIR_TMP
	CrowdHuman $GTR_DIR_TMP
	BDD100K $GTR_DIR_TMP
else
    echo "Error: TMPDIR not set"
    exit
fi
