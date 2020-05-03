#// python train.py --model_name "LSTMNet_base" --model_dir "./experiment/LSTMNet/base_model" --data_dir "./data/rml_data_5_600_with_high_snr.h5"
#// python train.py --model_name "LSTMNet_2xHidden256" --model_dir "./experiment/LSTMNet/wider_or_deeper/256x2" --data_dir "./data/rml_data_5_600_with_high_snr.h5"
#// python train.py --model_name "LSTMNet_2xHidden128" --model_dir "./experiment/LSTMNet/wider_or_deeper/128x2" --data_dir "./data/rml_data_5_600_with_high_snr.h5"
#! python train.py --model_name "CLDNN_base" --model_dir "./experiment/CLDNN/base_model" --data_dir "./data/rml_data_5_600_with_high_snr.h5" --gpu_id 2
python train.py --model_name "GRUNet_base" --model_dir "./experiment/GRUNet/base_model" --data_dir "./data/rml_data_5_600_with_high_snr.h5" --gpu_id 1
# python train.py --model_name "GRUNet_2xHidden256" --model_dir "./experiment/GRUNet/hidden_size/256x2" --data_dir "./data/rml_data_5_600_with_high_snr.h5" --gpu_id 1
# python train.py --model_name "GRUNet_3xHidden128" --model_dir "./experiment/GRUNet/hidden_size/128x3" --data_dir "./data/rml_data_5_600_with_high_snr.h5" --gpu_id 2
# python train.py --model_name "GRUNet_2xHidden128" --model_dir "./experiment/GRUNet/hidden_size/128x2" --data_dir "./data/rml_data_5_600_with_high_snr.h5" --gpu_id 2
python train.py --model_name "CLDNN_base" --model_dir "./experiment/CLDNN/base_model" --data_dir "./data/rml_data_5_600_with_high_snr.h5" --gpu_id 1
python train.py --model_name "CLDNN_kernel50x6" --model_dir "./experiment/CLDNN/kernel/50x6" --data_dir "./data/rml_data_5_600_with_high_snr.h5" --gpu_id 0
python train.py --model_name "CLDNN_kernel50x4" --model_dir "./experiment/CLDNN/kernel/50x4" --data_dir "./data/rml_data_5_600_with_high_snr.h5" --gpu_id 0
python train.py --model_name "CLDNN_kernel30x8" --model_dir "./experiment/CLDNN/kernel/30x8" --data_dir "./data/rml_data_5_600_with_high_snr.h5" --gpu_id 0