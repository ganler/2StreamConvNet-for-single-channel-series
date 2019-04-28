# 2StreamConvNet for single channel series

**See our latest code (here)[https://github.com/ganler/2StreamConvNet-for-single-channel-series/blob/master/%E5%A4%9A%E5%AA%92%E4%BD%93%E5%A4%A7%E4%BD%9C%E4%B8%9A%E5%AE%9E%E8%B7%B5%E6%8A%A5%E5%91%8A%E4%B8%8E%E4%BB%A3%E7%A0%81.ipynb] and lastest paper (here)[https://github.com/ganler/2StreamConvNet-for-single-channel-series/blob/master/Transfer-Learning-Apply-Image-Recognition-Models-to-Action-Recognition-with-Dynamic-Vision%20Sensor.pdf].**


> This repo is a **pytorch** implementation of 2 stream convolutional neural networks, which can be used in detached single channel frames series.
>
> By a student of Tongji University.
>
> Contact me at jaway.liu@gmail.com or just make an issue.

## Brief introduction

Two stream convNet is a famous CNN architecture series in action recognition. The former 2 stream convNet implementations are mainly based on RGB frame series(input of `3*len(series)`), which cannot use some strong pretrained model(e.g. ResNet-50) in image recognition(input of `3` channel only).

In some cases, we do action recognition with frames of 1 channels(gray pics, dynamic vision sensor(DVS) pics). If you want to test how pretrained image recognition models works on single channel action recognition, this repo will help you. But attention that the length of input single channel series must be 3(as RGB image has only 3 channels). I also provide few simple data to test if the code works.(At least, it works on my MacOS). To be honest, the code is not that hard to write, as PyTorch provided a user-friendly API.(The most tricky part is the dataloader and the arrangement of files.)

## Environment

- PyTorch 1.0.0 based on python 3.7；

- Numpy & Pandas & PIL.


## Pipeline

- Prepare the data(See more details in `Detail` part).

- Split the data into train/valid/test(Generate a .csv for each);

  > `python dvs_split_train_valid_test.py`

- Design the data loader to load the data;

- train spatial network(fine-tune ResNet/VGG);

  > `python motion_stream.py`

- train motion network(fine-tune ResNet/VGG);

  > ``python spatial_stream.py``

- Fusion(Average fusion);

  > `python avg_fusion.py`

## Details

#### The files arrangement:

![img](https://s2.ax1x.com/2019/01/20/kCTLqS.png)

#### Dataloader details:

I arrange the data like this[#]:

```
- DataSet/
	- class_0/
		- sample_0/
			- motion/
				... motion frames
			- spatial/
				... spatial frames
				(num(motion frames) should == num(spatial frames))
		- sample_1/
		...
		- sample_N/
	- class_1/
	...
	- class_N/
```

- `dvs_split_train_valid_test.py`

> For files arranged in # form, I split them into `train/valid/test` in `.csv` form which can be used in dataloader.
>
> `split_por` means the proportion of each.
>
> The `.csv`s generated looks like this:

![img](https://s2.ax1x.com/2019/01/20/kC7AZF.png)

> `s_frame_x` means the (x+1) channel of the spatial image we want to feed to the net.(We use 3 gray scale pic as the 3 channel of the input of the network.)
>
> `m_frame_x` means the (x+1) channel of the motion image we want to feed to the net.
>
> `tags`: just their labels.

- `spatial_and_motion_dataloader.py`

> In this file I designed the `sm_Dataset ` inherited from `torch.utils.data.Dataset`.
>
> The `__init__` function has a attribute "mode". If mode == `'spatial'/'motion'`, we load the spatial/motion data from the .csv. 

```python
img_0 = Image.open(self.channel_0[index]).convert('L') # 对于灰度图是'L'
img_1 = Image.open(self.channel_1[index]).convert('L')
img_2 = Image.open(self.channel_2[index]).convert('L')

if self.transform is not None:
    img_0 = self.transform(img_0)
    img_1 = self.transform(img_1)
    img_2 = self.transform(img_2)
    
img = torch.cat([img_0, img_1, img_2], 0)
```

> This is how I made the 3 channels into one "RGB-like" data.

- `spatial/motion_stream.py`

> Fine-tune the networks. The following models can be chosen.

```python
# MODEL, LOSS FUNC AND OPTIMISER
# resnet
# model = models.ResNet(pretrained=True)
model = models.resnet18(pretrained=True)
# model = models.resnet34(pretrained=True)
# model = models.resnet50(pretrained=True)

# vgg
# model = models.VGG(pretrained=True)
# model = models.vgg11(pretrained=True)
# model = models.vgg16(pretrained=True)
# model = models.vgg16_bn(pretrained=True)
```

> There're 2 modes for training you can select. Replace the comment to choose mode you want.
>
> **Mode 1**: Freeze the weights in ealier layers.
>
> **Mode 2**: Instead of freezing the weights in ealier layers, I just make their learning rate smaller.

> Of course you can replace the value of `num_class`(in my case it's 17) for your dataset.

```python
# TRAIN MODE 1 ================================
    model.load_state_dict(model_dict)
    # 至此fine-tune对应的结构已经搞定

    # 除了最后两层，其余都把梯度给冻结
    for para in list(model.parameters())[:-2]:
        para.requires_grad = False

    # 只训练最后2层
    optimizer = torch.optim.Adamax(params=[model.fc.weight, model.fc.bias], lr=learning_rate, weight_decay=1e-4)
# -------------================================

# # TRAIN MODE 2 ================================
#     ignored_params = list(map(id, model.parameters()[:-2]))
#     # fc3是net中的一个数据成员
#     base_params = filter(
#         lambda p: id(p) not in ignored_params,
#         model.parameters()
#     )
#     '''
#     id(x)返回的是x的内存地址。上面的意思是，对于在net.parameters()中的p，过滤掉'id(p) not in ignored_params'中的p。
#     '''
#
#     optimizer = torch.optim.Adamax(
#         [{'params': base_params},
#          {'params': model.fc3.parameters(), 'lr': learning_rate}],
#         1e-3, weight_decay=1e-4
#     )
# # -------------================================
```

> The output of the 2 streams(in one-hot form) will be released to `.txt`.
>
> We'll get 3 `.txt`:
>
> - `spatial_out.txt`
> - `motion_out.txt`
> - `label_out.txt`
>
> All this can be loaded by numpy. These data will can used in fusion. (Note that the `shuffle` param of dataloader should be 'False'. Cuz once you use `shuffle`, the data cannot be matched up)

- `avg_fusion.py`

> I think it's easy to understand.

```python
import numpy as np

motion_out = np.loadtxt('motion_out.txt')
spatial_out = np.loadtxt('spatial_out.txt')
label_out = np.loadtxt('label_out.txt')
fusion = motion_out+spatial_out

arg_mx = np.argmax(fusion, axis=0)
print(f">>> acc: {(arg_mx == label_out).mean()}")
```

## Problems I met

#### Input size and network structure:

> I finally decide to use constant 3 of the spatial/motion images(their channels are 1), to form an 'RGB-like' picture as the input of network like ResNet. And I just use some existing structures as the classifier.

## Reference

\[1\][Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional) 
