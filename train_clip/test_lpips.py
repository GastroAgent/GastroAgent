import lpips
import torch

# 初始化模型
loss_fn = lpips.LPIPS(net='alex', spatial=False).to(torch.device('cpu'))

# 加载图像并转换为张量
img0 = lpips.im2tensor(lpips.load_image('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/01.2018030100098_7-High-VAE/aug_image.jpg'))
img1 = lpips.im2tensor(lpips.load_image('/home/dalhxwlyjsuo/criait_tansy/project/EndoViT/our_eval_data/new_cropped-2004-2010-endovit/01.2018030100098_7-High-VAE/gen_aug_image.jpg'))

print(img0.shape)
print(img1.shape)

# 计算LPIPS损失
lpips_loss = loss_fn.forward(img0, img0)

print(f'LPIPS loss:')
print(lpips_loss)
print(lpips_loss.shape)