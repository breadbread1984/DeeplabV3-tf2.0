# DeeplabV3-tf2.0
This project implements the SOTA image segmentation algorithm deeplab V3+ with tensorflow2

## dataset preparation

download COCO2017 dataset from [here](https://cocodataset.org/). unzip directory train2017, val2017 and annotations. generate dataset with the following command.

```python
python3 create_dataset.py </path/to/train2017> </path/to/val2017> </path/to/annotations>
```

upon executing the script successfully, there will directory trainset and testset generated under the root directory of the source code.

## train with dataset

train with multiple GPU with executing command

```python
python3 train_eager_distributed.py
```

train with single GPU with executing command

```python
python3 train_eager.py
```

or 

```python
python3 train_keras.py
```


