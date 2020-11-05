import torch


def fgsm_classifier_attack(net, images, category_labels, texture_labels,
                           category_criterion, texture_criterion, epsilon):
    net.model.requires_grad = False
    images.requires_grad = True

    if images.grad is not None:
        images.grad.zero_()

    category_outputs, texture_outputs = net.model(images)
    category_loss = category_criterion(category_outputs, category_labels)
    texture_loss = texture_criterion(texture_outputs, texture_labels)
    loss = category_loss + texture_loss
    loss.backward()

    adversarial_images = images + epsilon * images.grad.sign()
    return torch.clamp(adversarial_images, 0, 1).detach()


def pgd_classifier_attack(net,
                          images,
                          category_labels,
                          texture_labels,
                          category_criterion,
                          texture_criterion,
                          epsilon=0.03,
                          k=8):
    if k <= 0:
        return images

    step_size = epsilon / k
    adversarial_images = images.clone()

    for i in range(k):
        adversarial_images = fgsm_classifier_attack(net,
                                                    adversarial_images,
                                                    category_labels,
                                                    texture_labels,
                                                    category_criterion,
                                                    texture_criterion,
                                                    epsilon=step_size)
        adversarial_images = torch.max(adversarial_images, images - epsilon)
        adversarial_images = torch.min(adversarial_images, images + epsilon)

    return adversarial_images
