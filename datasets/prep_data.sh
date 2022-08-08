GTR_DIR=~/GTR

MOT17 () {
	DATASETS_DIR=$1/datasets
	cd "$DATASETS_DIR" || exit
	if ! [[ -d mot/ ]]
	then
		mkdir mot/ && cd mot/ || exit
		if [[ -f $GTR_DIR/datasets/mot/MOT17.zip ]]; then
		    echo "Copying MOT17 dataset..."
		    cp $GTR_DIR/datasets/mot/MOT17.zip ./
		else
		    echo "Downloading MOT17 dataset..."
		    wget -q https://motchallenge.net/data/MOT17.zip
		fi
        echo "Extracting MOT17 dataset..."
		unzip -q MOT17.zip
        cd MOT17/ || exit
        ln -s train trainval
        cd $1 || exit
        echo "Converting MOT17 dataset format..."
        python tools/convert_mot2coco.py trainval
        python tools/convert_mot2coco.py test
    fi
}

CrowdHuman () {
	DATASETS_DIR=$1/datasets
    cd "$DATASETS_DIR" || exit
	if ! [[ -d crowdhuman/ ]]; then
	    CH_SUCCESS=0
	    mkdir crowdhuman/ && cd crowdhuman/ || exit
	    if [[ -d $GTR_DIR/datasets/crowdhuman/ ]]; then
	        echo "Copying CrowdHuman dataset..."
	        cp $GTR_DIR/datasets/crowdhuman/*.zip ./
	        cp $GTR_DIR/datasets/crowdhuman/*.odgt ./
	        if [[ $(ls -1 | wc -l) == 6 ]]; then
		        CH_SUCCESS=1
            else
                echo "Failed copying CrowdHuman dataset"
		        rm ./*
            fi
        fi
        if [[ $CH_SUCCESS == 0 ]]; then
            echo "Downloading CrowdHuman dataset..."
            cd ../
		    gdown -q --folder 1-N59uI5plTXEepaIasH3kpFW6SPKSVkQ
		    cd crowdhuman/ || exit
        fi
        if [[ $(ls -1 | wc -l) != 6 ]]; then
		    echo "Error: Incomplete CrowdHuman dataset"
		    exit
        fi
        echo "Extracting CrowdHuman dataset..."
		unzip -q CrowdHuman_train\* -d CrowdHuman_train/
		unzip -q CrowdHuman_val\* -d CrowdHuman_val/
		cd "$1" || exit
		echo "Converting CrowdHuman dataset format..."
		python tools/convert_crowdhuman_amodal.py
	fi
}

BDD100K () {
	DATASETS_DIR=$1/datasets
	cd "$DATASETS_DIR" || exit
	if ! [[ -d bdd/ ]]; then
		if [[ $HOSTNAME =~ "turing" || $HOSTNAME =~ "vista" ]]; then
		    if [[ -f /nas/vista-ssd02/users/jmathai/bdd100k_qdtrack_data.tar ]]; then
		        echo "Copying BDD100K dataset..."
		        cp /nas/vista-ssd02/users/jmathai/bdd100k_qdtrack_data.tar ./
		        echo "Extracting BDD100K dataset..."
		        tar -xf bdd100k_qdtrack_data.tar
		        mv data/ bdd/
		        mv bdd/bdd/ bdd/BDD100K/
		    else
		        echo "BDD100K dataset does not exist: /nas/vista-ssd02/users/jmathai/bdd100k_qdtrack_data.tar"
            fi
        fi
        if ! [[ -d bdd/ ]]; then
            mkdir bdd/ && cd bdd/ || exit
            echo "Downloading BDD100K dataset..."
            gdown -q 1u1DKH7Stk-YNRhhGHHJx_4lUH7sA_e6m
            curl -f -L http://dl.yf.io/bdd100k/mot20/ | grep -oe "images.*"\"$"" > bdd100k.txt
            sed -i 's#"##' bdd100k.txt && sed -i 's#^#http://dl.yf.io/bdd100k/mot20/#' bdd100k.txt
            cat bdd100k.txt | xargs -n 1 -P 3 wget -q
            for file in *; do
                if [[ $file =~ ".md5" ]]; then
                    if ! [[ $(md5sum "${file//.md5/}" | cut -d' ' -f 1) == $(cat $file) ]]; then
                        echo "Error: MD5 checksum does not match for" "${file//.md5/}"
                        exit
                    else
                        echo "$file" "checked"
                    fi
                fi
            done
            rm ./*.md5 ./*.txt
            if [[ $(ls -1 | wc -l) != 11 ]]; then
                echo "Error: Incomplete BDD100K dataset"
                exit
            fi
            echo "Extracting BDD100K dataset..."
            ls -1 | xargs -n 1 -P 0 unzip -q
            mv bdd100k/ BDD100K/
		    cd $1/tools || exit
		    echo "Converting BDD100K dataset format..."
		    python3 -m bdd100k.label.to_coco -m box_track -i $DATASETS_DIR/bdd/BDD100K/labels/box_track_20/train/ -o $DATASETS_DIR/bdd/BDD100K/labels/box_track_20/box_track_train_cocofmt.json
            python3 -m bdd100k.label.to_coco -m box_track -i $DATASETS_DIR/bdd/BDD100K/labels/box_track_20/val/ -o $DATASETS_DIR/bdd/BDD100K/labels/box_track_20/box_track_val_cocofmt.json
        fi
    fi
}

# Set TMPDIR if on iLab or Discovery
if [[ $HOSTNAME =~ iGpu || $HOSTNAME =~ iLab ]]; then
	export TMPDIR=/lab/tmpig8e/u/brian-data
elif [[ $HOSTNAME =~ "discovery" || $HOSTNAME =~ "hpc" || $HOSTNAME =~ [a-z][0-9][0-9]-[0-9][0-9] ]]; then
    export TMPDIR=/scratch1/briannlz
    module load gcc
    module load unzip
elif [[ $HOSTNAME =~ "turing" || $HOSTNAME =~ "vista" ]]; then
    :
else
    echo "Error: Unknown host: $HOSTNAME"
    exit
fi

# Download to data directory specified by $TMPDIR if set
if [[ -v TMPDIR ]]; then
	echo "Preparing data in TMPDIR=$TMPDIR"
	GTR_DIR_TMP=$TMPDIR/GTR
	if ! [[ -d $GTR_DIR_TMP/datasets ]]; then
		mkdir -p $GTR_DIR_TMP/datasets/
	fi
	rm -r $GTR_DIR_TMP/datasets/metadata/
    cp -r $GTR_DIR/datasets/metadata/ $GTR_DIR_TMP/datasets/metadata/
    rm -r $GTR_DIR_TMP/tools/
	cp -r $GTR_DIR/tools/ $GTR_DIR_TMP/tools/
	rm -r $GTR_DIR_TMP/models/
	cp -r $GTR_DIR/models/ $GTR_DIR_TMP/models/

	MOT17 $GTR_DIR_TMP
	CrowdHuman $GTR_DIR_TMP
	BDD100K $GTR_DIR_TMP
else
    echo "Error: TMPDIR not set, "
    exit
fi
