### Dictionary of models
model_dict = {
    "custom-v1"     :   custom_model.Cifar10CnnModel(),
    "resnet18-v0"   :   resnet_model.PreTrainedResNet()
}

### How to run the program

For running main.py
- You're at the same directory with the main.py file
- Set your python environment to have installed all the dependencies in requirements.txt


python main.py \
    --model [type of model] \
    --epochs [number of epochs] \
    --lr [learning rate] \
    --batch_size [batch size]

Default args
model       :   resnet18-v0 
epochs      :   30
lr          :   0.001
batch_size  :   32


### test run

without arguments for parameters
```
python3.10 main.py 
```

with arguments for parameters
```
python3.10 main.py \
    --model resnet18-v0 \
    --epochs 50 \
    --lr 0.001 \
    --batch_size 32
```
