import torchvision.models as models
from torchvision import transforms
import torch
import numpy as np
from torch import nn
from src.spatial_and_motion_dataloader import * # self-made

if __name__ == "__main__":
    key_word = 'spatial'

    # DEVICE
        # ########## !!! LOOK HERE !!! ############ #
    use_gpu = 0
        # ######################################### #
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    # PARAMETERS
    num_epoachs = 1
    batch_size = 5
    times4print = 100 / batch_size  # time for print (I print the info for every * batches)
    num_classes = 17
    classes = np.arange(num_classes)
    learning_rate = 0.01

    train_dataset = sm_dataset('train.csv', key_word, transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    valid_dataset = sm_dataset('valid.csv', key_word, transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))

    # LOADER
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               num_workers=2,
                                               shuffle=False)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              num_workers=2,
                                              shuffle=False)

# ===========CHOOSE THE MODELS===========
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

    pre_model = model

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    pretrained_dict = pre_model.state_dict()
    model_dict = model.state_dict()

    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)

# TRAIN MODE 1 ================================
    model.load_state_dict(model_dict)
    # 至此fine-tune对应的结构已经搞定

    # 除了最后两层，其余都把梯度给冻结
    for para in list(model.parameters())[:-2]:
        para.requires_grad = False

    # 只训练最后2层
    optimizer = torch.optim.Adamax(params=[model.fc.weight, model.fc.bias], lr=learning_rate, weight_decay=1e-4)
# -------------================================
#
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

    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()

    loss_func = nn.CrossEntropyLoss()

    # TRAIN
    total_steps = len(train_loader)
    for epoach in range(num_epoachs):
        loss_accumulation = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            loss = loss_func(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_accumulation += loss.item()
            if (i + 1) % times4print == 0:
                print(f"[{epoach+1}/{num_epoachs}]: -> [{i+1}/{total_steps}] -> loss: {loss_accumulation/times4print}")
                loss_accumulation = 0

    # TEST
    model.eval()

    label_cp = np.zeros(len(valid_loader))
    np_out = np.zeros((len(valid_loader), num_classes))
    with torch.no_grad():
        class_correct = list(0. for i in range(num_classes))
        class_total = class_correct.copy()
        for k, (imgs, labels) in enumerate(valid_loader):
            label_cp[k] = labels.numpy()

            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            np_out[k] = out.numpy()

            _, predicted = torch.max(out, 1)
            ans_batch = (predicted == labels).squeeze()
            for k, label in enumerate(labels):
                if ans_batch.item() == 1: # right
                    class_correct[label] += 1
                class_total[label] += 1
        if sum(class_total) != 0:
            print(f">>> FINAL ACCURACY: {100 * sum(class_correct)/sum(class_total)}% -> {class_correct}/{class_total}")
        for i in range(num_classes):
            if class_total[i] != 0:
                print(f">>> [{classes[i]}] : {100 * class_correct[i]/class_total[i]}% -> {class_correct[i]}/{class_total[i]}")

    np.savetxt(key_word+'_out.txt', np_out)
    np.savetxt('label_out.txt', label_cp)
    torch.save(model.state_dict(), key_word+'_stream_model.ckpt')