ENV_PATH=~/anaconda3/envs/gtr/bin/python

TEST_CUDA_PYTORCH () {
    echo Test CUDA and Pytorch
    $ENV_PATH -c "import torch; print('Pytorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print(torch.randn(5).cuda())"
}

TEST_D2 () {
    echo Test Detectron2
    D2_PATH=~/detectron2
    $ENV_PATH $D2_PATH/demo/demo.py \
    --input $D2_PATH/demo/input/*.jpg \
    --output $D2_PATH/demo/output/ \
    --config-file $D2_PATH/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
    --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
}

TEST_GTR () {
    echo Test GTR
    GTR_PATH=~/GTR
    cd $GTR_PATH || exit
    $ENV_PATH $GTR_PATH/demo.py \
    --config-file $GTR_PATH/configs/GTR_TAO_DR2101.yaml \
    --video-input $GTR_PATH/docs/yfcc_v_acef1cb6d38c2beab6e69e266e234f.mp4 \
    --output $GTR_PATH/output_demo/demo_yfcc.mp4 \
    --opts MODEL.WEIGHTS models/GTR_TAO_DR2101.pth
}

TEST_CUDA_PYTORCH
TEST_D2
TEST_GTR
