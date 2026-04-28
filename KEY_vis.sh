# =================================
# the .pth file to visulize the seg map
# =================================

# =================================
# python demo/demo.py \
#     --config-file xx \
#     --input xx \
#     --output xx \
#     --opts MODEL.WEIGHTS xx 

# python demo/demo_visual_gt.py \
#     --config-file xx \
#     --input xx \
#     --gt xx \
#     --output xx \
# =================================

# =================================
# DLRSD
# =================================
python demo/demo.py --config-file configs/vitb_384_DLRSD.yaml \
 --input  datasets/DLRSD/imgs/Airplane00.jpg \
  datasets/DLRSD/imgs/Airplane01.jpg \
  datasets/DLRSD/imgs/Airplane02.jpg \
  datasets/DLRSD/imgs/Airplane03.jpg \
  datasets/DLRSD/imgs/Airplane04.jpg \
  datasets/DLRSD/imgs/Airplane05.jpg \
  datasets/DLRSD/imgs/Airplane06.jpg \
  datasets/DLRSD/imgs/Airplane07.jpg \
  datasets/DLRSD/imgs/Airplane08.jpg \
  datasets/DLRSD/imgs/Airplane09.jpg \
  datasets/DLRSD/imgs/Airplane10.jpg \
  datasets/DLRSD/imgs/Airplane11.jpg \
  datasets/DLRSD/imgs/Airplane12.jpg \
  datasets/DLRSD/imgs/Airplane13.jpg \
 --output  vis_output_DLRSD\
 --opts MODEL.WEIGHTS  fx_download/0SAVEoutput_vitb_384_DLRSD_rotate_dino_remoteclip_3W_layer5/model_final.pth

python demo/demo_visual_gt.py --config-file configs/vitb_384_DLRSD.yaml \
 --input  datasets/DLRSD/imgs/Airplane00.jpg \
  datasets/DLRSD/imgs/Airplane01.jpg \
  datasets/DLRSD/imgs/Airplane02.jpg \
  datasets/DLRSD/imgs/Airplane03.jpg \
  datasets/DLRSD/imgs/Airplane04.jpg \
  datasets/DLRSD/imgs/Airplane05.jpg \
  datasets/DLRSD/imgs/Airplane06.jpg \
  datasets/DLRSD/imgs/Airplane07.jpg \
  datasets/DLRSD/imgs/Airplane08.jpg \
  datasets/DLRSD/imgs/Airplane09.jpg \
  datasets/DLRSD/imgs/Airplane10.jpg \
  datasets/DLRSD/imgs/Airplane11.jpg \
  datasets/DLRSD/imgs/Airplane12.jpg \
  datasets/DLRSD/imgs/Airplane13.jpg \
 --gt   datasets/DLRSD/D2masks/ \
 --output  gt_output_DLRSD\


# =================================
# iSAID
# =================================
python demo/demo.py --config-file configs/vitb_384_iSAID.yaml \
 --input  datasets/iSAID/imgs/P0002_1030_1286_1442_1698.png \
  datasets/iSAID/imgs/P0002_1030_1286_1648_1904.png \
  datasets/iSAID/imgs/P0002_1030_1286_1854_2110.png \
  datasets/iSAID/imgs/P0002_1030_1286_2266_2522.png \
  datasets/iSAID/imgs/P0002_1030_1286_2301_2557.png \
  datasets/iSAID/imgs/P0002_1442_1698_2266_2522.png \
  datasets/iSAID/imgs/P0002_1442_1698_2301_2557.png \
  datasets/iSAID/imgs/P0002_1648_1904_2266_2522.png \
  datasets/iSAID/imgs/P0002_1648_1904_2301_2557.png \
  datasets/iSAID/imgs/P0003_206_462_206_462.png \
  datasets/iSAID/imgs/P0003_206_462_412_668.png \
  datasets/iSAID/imgs/P0003_412_668_206_462.png \
  datasets/iSAID/imgs/P0003_412_668_412_668.png \
  datasets/iSAID/imgs/P0003_767_1023_618_874.png \
 --output  vis_output_iSAID\
 --opts MODEL.WEIGHTS  fx_download/0SAVEoutput_vitb_384_DLRSD_rotate_dino_remoteclip_3W_layer5/model_final.pth

python demo/demo_visual_gt.py --config-file configs/vitb_384_iSAID.yaml \
 --input  datasets/iSAID/imgs/P0000_1648_1904_3090_3346.png \
 --input  datasets/iSAID/imgs/P0002_1030_1286_1442_1698.png \
  datasets/iSAID/imgs/P0002_1030_1286_1648_1904.png \
  datasets/iSAID/imgs/P0002_1030_1286_1854_2110.png \
  datasets/iSAID/imgs/P0002_1030_1286_2266_2522.png \
  datasets/iSAID/imgs/P0002_1030_1286_2301_2557.png \
  datasets/iSAID/imgs/P0002_1442_1698_2266_2522.png \
  datasets/iSAID/imgs/P0002_1442_1698_2301_2557.png \
  datasets/iSAID/imgs/P0002_1648_1904_2266_2522.png \
  datasets/iSAID/imgs/P0002_1648_1904_2301_2557.png \
  datasets/iSAID/imgs/P0003_206_462_206_462.png \
  datasets/iSAID/imgs/P0003_206_462_412_668.png \
  datasets/iSAID/imgs/P0003_412_668_206_462.png \
  datasets/iSAID/imgs/P0003_412_668_412_668.png \
  datasets/iSAID/imgs/P0003_767_1023_618_874.png \
 --gt   datasets/iSAID/D2masks \
 --output  gt_output_iSAID\