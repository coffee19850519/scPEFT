from new import convert_file

convert_file(
    pt_filename="/public/home/wxy/scPEFT_reproduction-main/geneformer1/Pretrain_ckpts/geneformer-12L-30M-finetune/pytorch_model.bin",
    sf_filename="/public/home/wxy/scPEFT_reproduction-main/geneformer1/Pretrain_ckpts/geneformer-12L-30M-finetune/model.safetensors",
    discard_names=[]
)
