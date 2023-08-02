import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def earlystop(model, data, domain, target, step_size, epsilon, perturb_steps,tau,randominit_type,loss_fn,rand_init=True,omega=0,beta=0.5):
    '''
    The implematation of early-stopped PGD
    Following the Alg.1 in our FAT paper <https://arxiv.org/abs/2002.11242>
    :param step_size: the PGD step size
    :param epsilon: the perturbation bound
    :param perturb_steps: the maximum PGD step
    :param tau: the step controlling how early we should stop interations when wrong adv data is found
    :param randominit_type: To decide the type of random inirialization (random start for searching adv data)
    :param rand_init: To decide whether to initialize adversarial sample with random noise (random start for searching adv data)
    :param omega: random sample parameter for adv data generation (this is for escaping the local minimum.)
    :return: output_adv (friendly adversarial data) output_target (targets), output_natural (the corresponding natrual data), count (average backword propagations count)
    '''
    model.eval()

    K = perturb_steps
    output_domain = []
    output_adv = []
    output_natural = []
    output_target = []
    out_index =[]

    control = (torch.ones(len(domain)) * tau).cuda()
    index = torch.arange(0,len(domain),dtype=torch.int64).cuda()

    # Initialize the adversarial data with random noise
    if rand_init:
        if randominit_type == "normal_distribution_randominit":
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        if randominit_type == "uniform_randominit":
            iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
    else:
        iter_adv = data.cuda().detach()

    iter_clean_data = data.cuda().detach()
    iter_domain = domain.cuda().detach()
    iter_target = target.cuda().detach()
    iterout_index = index.cuda().detach()
    # iter_imgpath = imgPath.copy()
    output_iter_clean_data = model(data)[1]

    while K>0:
        iter_adv.requires_grad_()
        output = model(iter_adv)
        tar = output[0]
        output = output[1]
        pred = output.max(1, keepdim=True)[1]
        output_index = []
        iter_index = []

        # Calculate the indexes of adversarial data those still needs to be iterated
        for idx in range(len(pred)):
            if pred[idx] != iter_domain[idx]:
                if control[idx] == 0:
                    output_index.append(idx)
                else:
                    control[idx] -= 1
                    iter_index.append(idx)
            else:
                iter_index.append(idx)
        # Add adversarial data those do not need any more iteration into set output_adv
        if len(output_index) != 0:
            if len(output_domain) == 0:
                # incorrect adv data should not keep iterated
                output_adv = iter_adv[output_index].reshape(-1, 3, 128, 128).cuda()
                output_natural = iter_clean_data[output_index].reshape(-1, 3, 128, 128).cuda()
                output_domain = iter_domain[output_index].reshape(-1).cuda()
                output_target = iter_target[output_index].reshape(-1).cuda()
                out_index = iterout_index[output_index].reshape(-1).cuda()
                # output_imgpath = iter_imgpath[output_index].reshape(-1)
            else:
                # incorrect adv data should not keep iterated
                output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, 128, 128).cuda()), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, 128, 128).cuda()), dim=0)
                output_domain = torch.cat((output_domain, iter_domain[output_index].reshape(-1).cuda()), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).cuda()), dim=0)
                out_index = torch.cat((out_index, iterout_index[output_index].reshape(-1).cuda()), dim=0)

        # calculate gradient
        model.zero_grad()
        criterion=nn.CrossEntropyLoss().cuda()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_domain = criterion(output, iter_domain)
                loss_target = criterion(tar, iter_target)
                loss_adv = beta*loss_domain - (1-beta)*loss_target
                # loss_adv = loss_domain
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_domain = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
                loss_target = criterion(tar, iter_target)
                loss_adv = beta*loss_domain - (1-beta)*loss_target
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        # update iter adv
        if len(iter_index) != 0:
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            iter_clean_data = iter_clean_data[iter_index]
            iter_domain = iter_domain[iter_index]
            iter_target = iter_target[iter_index]
            iterout_index = iterout_index[iter_index]
            # iter_imgpath = iter_imgpath[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()

            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().cuda()
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, 0, 1)
        else:
            output_adv = output_adv.detach()
            return output_adv, output_domain, output_target, output_natural, out_index
        K = K-1

    if len(output_domain) == 0:
        output_domain = iter_domain.reshape(-1).squeeze().cuda()
        output_target = iter_target.reshape(-1).squeeze().cuda()
        out_index = iterout_index.reshape(-1).squeeze().cuda()
        output_adv = iter_adv.reshape(-1, 3, 128, 128).cuda()
        output_natural = iter_clean_data.reshape(-1, 3, 128, 128).cuda()
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, 128, 128)), dim=0).cuda()
        output_domain = torch.cat((output_domain, iter_domain.reshape(-1)), dim=0).squeeze().cuda()
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().cuda()
        out_index = torch.cat((out_index, iterout_index.reshape(-1)), dim=0).squeeze().cuda()
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, 128, 128).cuda()),dim=0).cuda()
    output_adv = output_adv.detach()
    return output_adv, output_domain, output_target, output_natural, out_index



def pgd(model, data, domain,target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        output_domain = output[0][1]
        output_target = output[0][0]
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_domain = nn.CrossEntropyLoss(reduction="mean")(output_domain, domain)
                loss_target = nn.CrossEntropyLoss(reduction="mean")(output_target, target)
                loss_adv = loss_domain - loss_target
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv