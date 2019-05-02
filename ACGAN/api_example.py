import shutil
import numpy as np
import torch
from models import *
import torchvision.utils as vutils

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_model():
    generator = Generator()
    generator.apply(weights_init)
    checkpoint = torch.load('./generate.pt')
    generator.load_state_dict(checkpoint['g_state_dict'])
    generator = generator.cuda()
    return generator

def edit(tar):
    allone = [1.0 for i in range(32)]
    allzero =[0.0 for i in range(32)]
    out = []
    for i in tar:
        tmp = []
        for j in range(4):
            if j == i:
                tmp += allzero
            else:
                tmp += allone
        out.append(tmp)
    out = torch.FloatTensor(out)
    return out.cuda()

def get_image(model,target):
    #target = np.array(target)
    tags = torch.FloatTensor(target).cuda()
    target = np.array(target)
    batch_size = target.shape[0]
    embedding = nn.Embedding(batch_size,128).cuda()
    max_tag = np.argmax(target,axis = 1)
  
    z = Variable(torch.FloatTensor(batch_size, 128))
    z = z.cuda()
    z.data.fill_(0.0)
    z.data.normal_(0, 1)

    tag = torch.LongTensor(max_tag).cuda()
    #embed = embedding(tag).view(batch_size,-1).cuda()
    #xx = z.mul(embed)
    factor = edit(max_tag)
    x = z.mul(factor)
    print(x.data)
    print(tags.data)
    
    rep = torch.cat((x, tags.clone()), 1)
    print(rep.data[0])
    fake = model.forward(rep).detach()[0]
    return fake

def get_by_label( model,index ):
    if index > 3:
        print("Index out of range!")
        return 0
    
    batch_size = 64
    num_tag = [np.random.randint(4) for _ in range(batch_size)]
    tag = [[1 if _ == num_tag[i] else 0 for _ in range(4)] for i in range(batch_size)]

    tag[0] = [0 for _ in range(4)]
    tag[0][index] = 1

    fake = get_image(model,tag)
    print(fake.data.shape)
    return fake.permute(1,2,0).numpy()

generator = load_model()
img = get_by_label(generator,1)
plt.imshow(img)