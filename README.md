This is a fork of https://github.com/FoundationVision/VAR

We propose a new model using VQWGAN instead of VQGAN in the tokenization part, and we will compare the results to the basic model.


## Rerun the results

- Install the requirements
  
```
pip install -r requirements.txt
```

- Download the cifar10 dataset

```
python dl_datasets.py
```

- Train the model

```
chmod +x launch.sh
./launch.sh
```

- Copy the generated checkpoint (containing the model)
```
cp local_output/ar-ckpt-best.pth var_d16.pth
```

- Generate many images

```
mkdir generated_images
python generate_images.py
```

- Calculate Inception Score (IS)

```
python calculate_is.py
```