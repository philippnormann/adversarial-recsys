import torch
import torch.nn as nn

from tqdm import tqdm
from torch import optim

from src.helpers import cos_dist, to_tanh_space, to_original_space


def fgsm_sim_attack(net, target_vecs, attack_images, epsilon):
    net.model.requires_grad = False
    attack_images.requires_grad = True

    if attack_images.grad is not None:
        attack_images.grad.zero_()

    attack_vecs = net.model(attack_images)

    loss = nn.functional.cosine_similarity(target_vecs, attack_vecs).sum()
    loss.backward()

    adversarial_images = attack_images + epsilon * attack_images.grad.sign()
    return torch.clamp(adversarial_images, 0, 1).detach()


def pgd_sim_attack(net, target_vecs, attack_images, epsilon=0.03, k=32):
    if k <= 0:
        return attack_images

    step_size = epsilon / k
    adversarial_images = attack_images.clone()
    t = tqdm(total=k, leave=False)

    for i in range(k):
        adversarial_images = fgsm_sim_attack(net,
                                             target_vecs,
                                             adversarial_images,
                                             epsilon=step_size)
        adversarial_images = torch.max(adversarial_images,
                                       attack_images - epsilon)
        adversarial_images = torch.min(adversarial_images,
                                       attack_images + epsilon)

        attack_vecs = net.model(adversarial_images)
        dist = cos_dist(target_vecs, attack_vecs).sum()
        t.set_description("dist: {:.4f}".format(dist))
        t.update()

    return adversarial_images


def cw_sim_attack(net,
                  target_vecs,
                  attack_images,
                  epsilon=0.03,
                  n=1000,
                  lr=5e-3,
                  c=2e+1):
    device = attack_images.device
    modifier = torch.zeros(attack_images.size()).float().to(device)

    net.model.requires_grad = False
    modifier.requires_grad = True

    t = tqdm(total=n, leave=False)

    for i in range(n):
        tanh_images = to_tanh_space(attack_images)
        adversarial_images = to_original_space(modifier + tanh_images)

        attack_vecs = net.model(adversarial_images)
        delta = torch.abs(adversarial_images - attack_images)

        loss1 = cos_dist(target_vecs, attack_vecs).sum()
        loss2 = torch.clamp(delta - epsilon, min=0.0).sum()

        loss = c * loss1 + loss2
        optimizer = optim.Adam([modifier], lr=lr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t.set_description("loss: {:.4f}".format(loss))
        t.update()

    return adversarial_images
