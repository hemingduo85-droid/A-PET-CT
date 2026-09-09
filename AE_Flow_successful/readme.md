train.py:选择训练或者选择评估生成权重和指标到checkpoints，生成.npz文件到paper_figures中用于绘图；（目前AE方法是正确流程）
infer.py:加载权重，并生成热力图到paper_figures。
paper_figures:加载各个方法.npz,生成全方法对比图。


训练命令：nohup python train.py --tracer FDG  --modalities ct,pet  --device cuda:2> fdg_ct_pet_train.log 2>&1 &

 nohup python train.py --tracer PSMA  --modalities ct,pet  --device cuda:3  > psma_ct_pet_train.log 2>&1 &
