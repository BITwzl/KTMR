# KTMR
Official code for paper ***Deep Learning based MRI Reconstruction with Transformer***

Put the dataset in proper paths and modify the configuration JSON file.

## Train
```cmd
python main_train_psnr.py --opt options/sample.json
```

## Test
```cmd
python main_test_psnr.py --opt options/sample.json
```