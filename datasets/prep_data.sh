GTR_DIR=~/GTR

MOT17() {
	DATASETS_DIR=$1/datasets
	cd "$DATASETS_DIR" || exit
	if ! [[ -d mot/ ]]; then
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
        cd "$1" || exit
        echo "Converting MOT17 dataset format..."
        python tools/convert_mot2coco.py trainval
        python tools/convert_mot2coco.py test
    else
        echo "mot/ already exists"
    fi
}

CrowdHuman() {
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
	else
	    echo "crowdhuman/ already exists"
    fi
}

BDD100K() {
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
		    cd "$1/tools" || exit
		    echo "Converting BDD100K dataset format..."
		    python3 -m bdd100k.label.to_coco -m box_track -i $DATASETS_DIR/bdd/BDD100K/labels/box_track_20/train/ -o $DATASETS_DIR/bdd/BDD100K/labels/box_track_20/box_track_train_cocofmt.json
            python3 -m bdd100k.label.to_coco -m box_track -i $DATASETS_DIR/bdd/BDD100K/labels/box_track_20/val/ -o $DATASETS_DIR/bdd/BDD100K/labels/box_track_20/box_track_val_cocofmt.json
        fi
    else
        echo "bdd/ already exists"
    fi
}

COCO2017() {
	DATASETS_DIR=$1/datasets
	cd "$DATASETS_DIR" || exit
	if ! [[ -d coco/ ]]; then
		mkdir coco/ && cd coco/ || exit
		if [[ $HOSTNAME =~ iGpu || $HOSTNAME =~ iLab ]]; then
	        if [[ -d /lab/tmpig8e/u/brian-data/COCO2017/train2017/ ]]; then
	            echo "Creating soft link for COCO train2017..."
                ln -s /lab/tmpig8e/u/brian-data/COCO2017/train2017/ train2017
            fi
            if [[ -d /lab/tmpig8e/u/brian-data/COCO2017/val2017/ ]]; then
                echo "Creating soft link for COCO val2017..."
                ln -s /lab/tmpig8e/u/brian-data/COCO2017/val2017/ val2017
            fi
            if [[ -d /lab/tmpig8e/u/brian-data/COCO2017/annotations/ ]]; then
                echo "Creating soft link for COCO annotations..."
                ln -s /lab/tmpig8e/u/brian-data/COCO2017/annotations/ annotations
            fi
            if [[ $(ls -1 | wc -l) == 3 ]]; then
		        return
            else
                rm ./*
            fi
        fi
		if [[ -f $GTR_DIR/datasets/coco/train2017.zip ]]; then
		    echo "Copying COCO train2017..."
		    cp $GTR_DIR/datasets/coco/train2017.zip ./
		else
		    echo "Downloading COCO train2017..."
		    wget -q http://images.cocodataset.org/zips/train2017.zip
		fi
		if [[ -f $GTR_DIR/datasets/coco/val2017.zip ]]; then
		    echo "Copying COCO val2017..."
		    cp $GTR_DIR/datasets/coco/val2017.zip ./
		else
		    echo "Downloading COCO val2017..."
		    wget -q http://images.cocodataset.org/zips/val2017.zip
		fi
		if [[ -f $GTR_DIR/datasets/coco/annotations_trainval2017.zip ]]; then
		    echo "Copying COCO annotations_trainval2017..."
		    cp $GTR_DIR/datasets/coco/annotations_trainval2017.zip ./
		else
		    echo "Downloading COCO annotations_trainval2017..."
		    wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
		fi
        echo "Extracting COCO2017 dataset..."
        ls -1 | xargs -n 1 -P 0 unzip -q
    else
        echo "coco/ already exists"
    fi
}

LVIS() {
    DATASETS_DIR=$1/datasets
	cd "$DATASETS_DIR" || exit
	if ! [[ -d lvis/ ]]; then
	    mkdir lvis/ && cd lvis/ || exit
	    if [[ -f $GTR_DIR/datasets/lvis/lvis_v1_train.json.zip ]]; then
		    echo "Copying LVIS train annotations..."
		    cp $GTR_DIR/datasets/lvis/lvis_v1_train.json.zip ./
		else
		    echo "Downloading LVIS train annotations..."
		    wget -q https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
		fi
		if [[ -f $GTR_DIR/datasets/lvis/lvis_v1_val.json.zip ]]; then
		    echo "Copying LVIS val annotations..."
		    cp $GTR_DIR/datasets/lvis/lvis_v1_val.json.zip ./
		else
		    echo "Downloading LVIS val annotations..."
		    wget -q https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
		fi
		echo "Extracting LVIS dataset..."
        ls -1 | xargs -n 1 -P 0 unzip -q
        cd "$1" || exit
        echo "Merging LVIS and COCO2017 annotations..."
        python tools/merge_lvis_coco.py
	else
	    echo "lvis/ already exists"
    fi
}

TAO() {
    DATASETS_DIR=$1/datasets
	cd "$DATASETS_DIR" || exit
	if ! [[ -d tao/ ]]; then
	    TAO_SUCCESS=0
	    mkdir tao/ && cd tao/ || exit
	    if [[ -d $GTR_DIR/datasets/tao/ ]]; then
	        echo "Copying TAO dataset..."
	        cp $GTR_DIR/datasets/tao/*.zip ./
	        if [[ $(ls -1 | wc -l) -ge 2 ]]; then
		        TAO_SUCCESS=1
            else
                echo "Failed copying TAO dataset"
		        rm ./*
            fi
        fi
        if [[ $TAO_SUCCESS == 0 ]]; then
            echo "Downloading TAO dataset..."
            #wget -q "https://motchallenge.net/data/1-TAO_TRAIN.zip"
            wget -q "https://motchallenge.net/data/2-TAO_VAL.zip"
            #wget -q "https://motchallenge.net/data/3-TAO_TEST.zip"
            #wget -q "https://motchallenge.net/data/1_AVA_HACS_TRAIN_67214e7e7fc77341d6eb3bc54d4d3e68.zip"
            wget -q "https://motchallenge.net/data/2_AVA_HACS_VAL_67214e7e7fc77341d6eb3bc54d4d3e68.zip"
            #wget -q "https://motchallenge.net/data/3_AVA_HACS_TEST_67214e7e7fc77341d6eb3bc54d4d3e68.zip"
            if [[ $(ls -1 | wc -l) -lt 2 ]]; then
		        echo "Error: Incomplete TAO dataset, go to MOT website for TAO AVA and HACS dataset"
		        exit
            fi
        fi
        echo "Extracting TAO dataset..."
        ls -1 | xargs -n 1 -P 0 unzip -q
        cd "$1" || exit
        echo "Downloading TAO annotations"
        python tools/tao/download/download_annotations.py "$DATASETS_DIR/tao/" --split train
        python tools/tao/download/verify.py "$DATASETS_DIR/tao/" --split train
        echo "Processing TAO dataset..."
        python tools/move_tao_keyframes.py --gt datasets/tao/annotations/validation.json --img_dir datasets/tao/frames --out_dir datasets/tao/keyframes
        python tools/create_tao_v1.py datasets/tao/annotations/validation.json
	else
	    echo "tao/ already exists"
    fi
}

# Set TMPDIR if on iLab or Discovery
if [[ $HOSTNAME =~ iGpu || $HOSTNAME =~ iLab ]]; then
	export TMPDIR=/lab/tmpig8b/u/brian-data
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
    conda init bash
    conda activate gtr
	echo "Preparing data in TMPDIR=$TMPDIR"
	GTR_DIR_TMP=$TMPDIR/GTR
	if ! [[ -d $GTR_DIR_TMP/datasets ]]; then
		mkdir -p $GTR_DIR_TMP/datasets/
	fi
	rm -rf $GTR_DIR_TMP/datasets/metadata/
    cp -r $GTR_DIR/datasets/metadata/ $GTR_DIR_TMP/datasets/metadata/
    rm -rf $GTR_DIR_TMP/tools/
	cp -r $GTR_DIR/tools/ $GTR_DIR_TMP/tools/
	rm -rf $GTR_DIR_TMP/models/
	cp -r $GTR_DIR/models/ $GTR_DIR_TMP/models/

    if [[ $# -eq 1 ]]; then
        echo "preparing $1 only"
        $1 $GTR_DIR_TMP
        exit
    fi
	MOT17 $GTR_DIR_TMP
	CrowdHuman $GTR_DIR_TMP
	BDD100K $GTR_DIR_TMP
	COCO2017 $GTR_DIR_TMP
	LVIS $GTR_DIR_TMP
	TAO $GTR_DIR_TMP
	echo "Done prepare datasets"
else
    echo "Error: TMPDIR not set"
    exit
fi
