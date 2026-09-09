### [Anomaly Detection for Medical Images Using Teacher-Student Model with Skip Connections and Multi-scale Anomaly Consistency](https://ieeexplore.ieee.org/document/10540605)

M Liu, Y Jiao, J Lu, H Chen - IEEE Transactions on Instrumentation and Measurement,
Anomaly Detection for Medical Images Using Teacher–Student Model With Skip Connections and Multiscale Anomaly Consistency, 2024

Check the paper in https://ieeexplore.ieee.org/abstract/document/10540605

Usage:

    # FDG: train 30 epochs, test once at the end, save only epoch 30.
    python main.py --mode train --dataset fdg --cuda 0

    # PSMA: same logic, dataset path is selected by --dataset.
    python main.py --mode train --dataset psma --cuda 6

    # Direct test with the automatically inferred checkpoint name.
    python main.py --mode test --dataset fdg --cuda 0

    # Equivalent explicit CUDA device form.
    python main.py --mode train --dataset fdg --device cuda:6

    # Save the first 20 heatmaps during final evaluation/test.
    python main.py --mode test --dataset fdg --heatmap_count 20

    # Save every heatmap.
    python main.py --mode test --dataset fdg --save_all_heatmaps

    # PET/CT input modes:
    #   pseudo_rgb: [CT, PET, PET] original-image pseudo RGB, default.
    #   dual: original two-channel CT+PET input.
    #   single: run each modality separately.
    python main.py --dataset fdg --modalities ct pet --input_mode pseudo_rgb
    python main.py --dataset fdg --modalities ct pet --input_mode dual
    python main.py --dataset fdg --modalities ct pet --input_mode single
     
Introduction of the code:

    1. dataset.py for loading the training and testing dataset.
    
    2. encoder.py for the encoder of the model which is pretrained.
    
    3. decoder.py which is the opposite of the structure of the encoder.
    
    4. loss_function.py as Figure 2(b) showed in the paper.
    
    5. eval_func.py for some useful functions.
    
    6. main.py for training and testing the model

    7. data.zip for the example dataset

Pretrained model on head_ct dataset:

    https://drive.google.com/file/d/1WKwX1xgPs7UpmWYcgBuVnZwbTSEShgli/view?usp=sharing
