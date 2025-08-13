# Weights-based BFA
# Adapted from https://github.com/elliothe/Neural_Network_Weight_Attack/blob/520643297ed7d0c7940128e70d4488cfc46a0845/attack/BFA.py

from typing import Iterable
import numpy as np
import operator
import math
import struct
import copy
from functional import seq
from tqdm import tqdm

import torch
import torch.nn as nn

import attacksim
import dataman
from support import torchdig

class WBBFA:

    MEM_TEMPLATE_VEC_LEN = attacksim.PAGE_SIZE // 4

    def __init__(self, ktop_weights=10, max_iters=20, memmodel: attacksim.MemoryModel = None) -> None:
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.ktop_weights = ktop_weights
        self.max_iters = max_iters

        self.templates_vecs = None
        if memmodel:
            self.templates_vecs = [
                self.gen_32bit_mem_template_vec(
                    filter(lambda t: t.direction == i, memmodel.templates)
                ) for i in range(2)
            ]

    @staticmethod
    def gen_32bit_mem_template_vec(templates: Iterable[attacksim.BitFlip]):
        '''Generates a (page_size/4,) vector representing the flippable bits
        according to the templates. Each 4 bytes (32 bits) are grouped as one
        element.
        The templates should have the same flip direction.'''
        # Start with a (page_size/4, 32) matrix of zeros
        assert attacksim.PAGE_SIZE % 4 == 0
        masks = torch.zeros((WBBFA.MEM_TEMPLATE_VEC_LEN, 32), dtype=torch.uint8)
        for t in templates:
            masks[t.in_page_offset // 4, t.in_page_offset % 4 * 8 + t.bitidx] = 1
        # Return a float vector (doesn't matter as it's expanded when used)
        return WBBFA.reconstruct_f32_tensor(masks.T)

    @staticmethod
    def _apply_mem_template_masks(w_ex, mask_flippable, tvec_this_dir, flip_direction):
        '''Returns a new mask_flippable with the memory templates (for the
        given flip direction) applied.
        w_ex: (32, n) expanded form of the weights
        mask_flippable: (32, n) original mask
        tvec_this_dir: (n,) vector of templates (for the given flip direction),
        expressed as 32-bit values
        flip_direction: 1 for 0 -> 1'''
        assert w_ex.ndim == 2 and w_ex.shape[0] == 32
        assert mask_flippable.ndim == 2 and mask_flippable.shape[0] == 32
        assert len(tvec_this_dir) == w_ex.shape[1] == mask_flippable.shape[1]
        assert flip_direction in (0, 1)

        # We need to ensure mask_flippable for bits not for this flip direction
        # are not touched.
        mask_applicable = w_ex == 1 - flip_direction
        mask_keep = ~mask_applicable
        t_ex_a = WBBFA.expand_f32_tensor(tvec_this_dir).bool() & mask_applicable
        updater = t_ex_a | mask_keep
        return (mask_flippable.bool() & updater).type_as(mask_flippable)

    def maybe_apply_memory_templates(self, w_ex, mask_flippable, tvec_idxs: Iterable[int]):
        '''A wrapper for _apply_mem_template_masks that applies the templates
        selected by tvec_idxs.'''
        if self.templates_vecs is None:
            return mask_flippable
        mask_flippable = self._apply_mem_template_masks(
            w_ex, mask_flippable, self.templates_vecs[0][tvec_idxs], 0
        )
        mask_flippable = self._apply_mem_template_masks(
            w_ex, mask_flippable, self.templates_vecs[1][tvec_idxs], 1
        )
        return mask_flippable

    @staticmethod
    def get_bit_grads(wgrad):
        '''Returns the per-bit gradient matrix (dtype_nbits x len(wgrad)) of a
        1-d float grad tensor.'''
        assert wgrad.dtype == torch.float32
        assert wgrad.ndim == 1

        partial_bcoeffs = torch.tensor([
            # Sign bit
            -2,
            # Exponent bits
            *(math.log(2) * torch.pow(2, torch.arange(7, -1, -1)).float()),
            # Mantissa bits - set to 0 because they're too small
            *(0 for _ in range(23)),
        ]).unsqueeze(1).float()
        # Each gradient still has a wgrad part left out
        bcoeffs = partial_bcoeffs * wgrad

        return bcoeffs * wgrad

    @staticmethod
    def expand_f32_tensor(t):
        '''Expands a (n,) float32 tensor into a (32, n) binary matrix.'''
        assert t.dtype == torch.float32
        assert t.ndim == 1
        return torch.tensor(
            seq(t)
            .map(lambda x: struct.pack('>f', x))  # Convert to bytes objects
            .map(lambda bs: int.from_bytes(bs, 'big'))  # Convert to ints
            # Convert to lists of digits
            .map(lambda i: [int(d) for d in bin(i)[2:].zfill(32)])
            .to_list()
        ).T

    @staticmethod
    def reconstruct_f32_tensor(t):
        '''Reconstructs a (32, n) binary matrix into a (n,) float32 tensor.'''
        assert t.ndim == 2
        assert t.shape[0] == 32
        return torch.tensor(
            seq(t.T.tolist())
            .map(lambda bits: int(''.join(map(str, bits)), 2))  # Convert to ints
            .map(lambda i: struct.pack('>I', i))  # Convert to bytes objects
            .map(lambda bs: struct.unpack('>f', bs)[0])  # Convert to floats
            .to_list()
        )

    @staticmethod
    def get_acc(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                output = model(x)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total

    @staticmethod
    def get_sus_score_range(model, train_loader):
        def get_range(mean, max, min):
            # TODO: We consider <= 0 scores as bad for now
            extend_coeff = 0.3
            ret = [min - (mean - min) * extend_coeff, max + (max - mean) * extend_coeff]
            ret[0] = np.maximum(ret[0], 1e-6)
            # print(f'{mean=}, {max=}, {min=}, {ret=}')
            return ret

        model.eval()
        sus_scores = []
        for x, y in train_loader:
            # x, y = x.to('cuda'), y.to('cuda')
            sus_scores.append(model.calc_sus_score(x).item())
        sus_scores = np.array(sus_scores)
        return get_range(sus_scores.mean(), sus_scores.max(), sus_scores.min())

    def in_layer_attack(
        self, m: nn.Module, n_bits2flip: int, ktop_weights=None, minimise_loss=False,
        evasion_mask=None, random_flips=False
    ):
        '''In-layer search and attack.'''
        # 1. flatten the gradient tensor to perform topk
        if ktop_weights is None:
            ktop_weights = m.weight.grad.numel()

        wgrads = m.weight.grad.detach().view(-1)
        if evasion_mask is not None:
            # Zero out the grads of the weights we don't want to flip (more later)
            wgrads *= 1. - evasion_mask.view(-1)
        if not random_flips:
            wgrads_abs = wgrads.abs()
            _, topk_wgrad_idxs = wgrads_abs.topk(ktop_weights, sorted=False)
        else:
            topk_wgrad_idxs = torch.randperm(wgrads.numel())[:ktop_weights]
        # print(f'{topk_wgrad_idxs=}')
        # update the b_grad to its signed representation
        topk_wgrad = wgrads[topk_wgrad_idxs]

        # 2. create the b_grad matrix in shape of [N_bits, k_top] (chain rule)
        topk_w_bgrad = self.get_bit_grads(topk_wgrad)
        # print(f'{topk_w_bgrad=}')

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        # Indicates the directions to move the weight bits toward
        topk_w_bgrad_sign = (topk_w_bgrad.sign() +
                             1) * 0.5  # zero -> negative, one -> positive
        # print(f'{topk_w_bgrad_sign=}')
        # "Expands" each weight to a binary column vector
        # Shape: [N_bits, k_top], binary matrix
        topk_w = m.weight.detach().view(-1)[topk_wgrad_idxs]
        topk_w_ex = self.expand_f32_tensor(topk_w)
        # print(f'{topk_w_ex=}')
        # E.g., if a weight bit is 1 and we still want it to increase (sign=1), then it's unflippable
        bgrad_mask_flippable = topk_w_ex ^ topk_w_bgrad_sign.short()
        # print(f'{bgrad_mask_flippable=}')

        if minimise_loss:
            bgrad_mask_flippable = (~bgrad_mask_flippable.bool()).short()

        # Apply memory templates - this assumes the weights are page aligned
        bgrad_mask_flippable = self.maybe_apply_memory_templates(
            topk_w_ex, bgrad_mask_flippable, topk_wgrad_idxs % WBBFA.MEM_TEMPLATE_VEC_LEN
        )

        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        # Sets the grads of unflippable bits to zero so we won't select those bits
        topk_w_bgrad *= bgrad_mask_flippable.float()
        # print(f'new {topk_w_bgrad=}')

        # Disallow flipping bits whose gradient is zero (e.g., due to evasion mask)
        topk_w_bgrad *= topk_w_bgrad.sign().abs()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        # abs() is fine because we already zeroed out the unflippable bits, and the remaining bits
        # are guaranteed to move towards the wanted direction if flipped
        grad_max = topk_w_bgrad.abs().max()
        if grad_max.item() == 0:
            # print('No bit to flip')
            return m.weight.detach().clone()

        _, flipped_bits_idxs = topk_w_bgrad.abs().view(-1).topk(n_bits2flip, sorted=False)
        mask_flips = topk_w_bgrad.clone().view(-1).zero_()

        mask_flips[flipped_bits_idxs] = 1
        mask_flips = mask_flips.view(topk_w_bgrad.size())

        # 6. update the weight in the original weight tensor
        new_topk_w_ex = topk_w_ex ^ mask_flips.short()
        new_topk_w = self.reconstruct_f32_tensor(new_topk_w_ex)
        new_w = m.weight.detach().clone().view(-1)
        new_w[topk_wgrad_idxs] = new_topk_w
        return new_w.view(m.weight.size())

    def cross_layer_attack(
        self, model, x, y,
        evasion_masks=None, max_nbits2flip=20, random_flips=False
    ):
        '''Runs one iteration of progressive bit search and returns the
        attacked model.
        evasion_masks specify the weight values not (1.0) to flip.
        '''
        model = copy.deepcopy(model)
        loss_dict = {}  # Per-layer loss

        model.eval()

        # 1. perform the inference w.r.t given data and y
        output = model(x)
        loss = self.loss_fn(output, y)
        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        loss.backward()
        # init the loss_max to enable the while loop
        loss_max = loss.item()

        # 3. for each layer flip #bits = self.bits2flip (cross-layer search)
        attacked_weights = {}
        for n_bits2flip in range(1, max_nbits2flip + 1):

            # iterate all the conv and linear layer
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    clean_weight = module.weight.data.detach()
                    attacked_weight = self.in_layer_attack(
                        module, n_bits2flip, self.ktop_weights,
                        evasion_mask=evasion_masks[name] if evasion_masks else None,
                        random_flips=random_flips
                    )
                    # change the weight to attacked weight and get loss
                    module.weight.data = attacked_weight
                    attacked_weights[name] = attacked_weight
                    output = model(x)
                    loss_dict[name] = self.loss_fn(output, y).item()
                    # change the weight back to the clean weight
                    module.weight.data = clean_weight

            # after going through all the layer, now we find the layer with max loss
            max_loss_module = max(loss_dict.items(), key=operator.itemgetter(1))[0]
            loss_max = loss_dict[max_loss_module]

            if not (loss_max <= loss.item()):  # As loss_max may become NaN, etc.
                break

        # 4. if the loss_max does lead to the degradation compared to the loss,
        # then change that layer's weight without putting back the clean weight
        if not (loss_max <= loss.item()):  # As loss_max may become NaN, etc.
            for name, module in model.named_modules():
                if name == max_loss_module:
                    print(name, n_bits2flip, loss.item(), loss_max)
                    module.weight.data = attacked_weights[name]
                    break
        else:
            n_bits2flip = 0  # Signal caller

        return model, n_bits2flip

    def attack(self, model, train_loader, test_loader, random_flips=False, sus_score_range=None):
        is_protected = isinstance(model, torchdig.DIGProtectedModule)
        orig_model = model
        orig_acc = self.get_acc(model, test_loader)
        print(f'Orig acc.: {orig_acc:.2%}')
        if is_protected:
            sus_score_range = sus_score_range or self.get_sus_score_range(model, train_loader)
            print(f'Orig sus score range: {sus_score_range}')

        nflipped = 0
        evasion_frac = 0.
        max_evasion_frac = 1.
        evasion_frac_step = 0.05
        evasion_masks = None
        x, y = next(iter(train_loader))
        for i in tqdm(range(self.max_iters)):

            # If protected by DIG, calculate the weights to not flip
            if is_protected and evasion_frac > 0.:
                model.calc_sus_score(x).backward()
                # Find the weights with top x percent of abs grads
                all_layers_grads = torch.cat([
                    m.weight.grad.detach().abs().view(-1)
                    for _n, m in model.named_modules()
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)
                ]).view(-1)
                grad_threshold = np.quantile(all_layers_grads.numpy(), 1. - evasion_frac).item()
                evasion_masks = {
                    n: (m.weight.grad.detach().abs() >= grad_threshold).float()
                    for n, m in model.named_modules()
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)
                }
                print(f'Masked {sum([int(x.sum()) for x in evasion_masks.values()])} weights')

            new_model, iter_nflipped = self.cross_layer_attack(
                model, x, y, evasion_masks=evasion_masks, random_flips=random_flips
            )
            if iter_nflipped == 0:
                # E.g., if there're too few memory templates
                break

            if is_protected:
                sus_score = new_model.calc_sus_score(x).item()
                print(f'Sus score after flipping: {sus_score:.2f}')
                if not (sus_score_range[0] <= sus_score <= sus_score_range[1]):
                    # If detected by DIG, revert the attack and raise evasion_frac
                    if evasion_frac >= max_evasion_frac:
                        print(f'DIG detected! Evasion frac. already at max')
                        break
                    evasion_frac = min(evasion_frac + evasion_frac_step, max_evasion_frac)
                    print(f'DIG detected! Evasion frac. increased to {evasion_frac:.2%}')
                    continue

            model = new_model
            nflipped += iter_nflipped
            acc = self.get_acc(model, test_loader)
            print(f'Acc. after flipping: {acc:.2%}')

            if (orig_acc - acc) / orig_acc >= 0.03:  # Goal: 3% drop
                print(f'Done! Flipped {nflipped} bits')
                return model, nflipped, evasion_frac

        print(f'Failed after {i+1} iterations')
        return orig_model, 0, evasion_frac

if __name__ == '__main__':
    import modman
    import dataman
    import utils

    for bi in [
        utils.BinaryInfo('tvm', 'main', 'resnet50', 'CIFAR10', 'cc2', 'gn1', True, 3),
        # utils.BinaryInfo('tvm', 'main', 'resnet50', 'MNISTC', 'cc2', 'gn1', True, 3),
        # utils.BinaryInfo('tvm', 'main', 'resnet50', 'FashionC', 'cc2', 'gn1', True, 3),
        utils.BinaryInfo('tvm', 'main', 'resnet50', 'ImageNet', 'cc2', 'gn1', True, 3),
        # utils.BinaryInfo('tvm', 'main', 'googlenet', 'CIFAR10', 'cc2', 'gn1', True, 3),
        # utils.BinaryInfo('tvm', 'main', 'googlenet', 'MNISTC', 'cc2', 'gn1', True, 3),
        # utils.BinaryInfo('tvm', 'main', 'googlenet', 'FashionC', 'cc2', 'gn1', True, 3),
        utils.BinaryInfo('tvm', 'main', 'googlenet', 'ImageNet', 'cc2', 'gn1', True, 3),
        # utils.BinaryInfo('tvm', 'main', 'densenet121', 'CIFAR10', 'cc2', 'gn1', True, 3),
        # utils.BinaryInfo('tvm', 'main', 'densenet121', 'MNISTC', 'cc2', 'gn1', True, 3),
        # utils.BinaryInfo('tvm', 'main', 'densenet121', 'FashionC', 'cc2', 'gn1', True, 3),
        utils.BinaryInfo('tvm', 'main', 'densenet121', 'ImageNet', 'cc2', 'gn1', True, 3),
    ]:
        print(f'Attacking {bi.model_name} on {bi.dataset} -----')

        tmod = modman.get_torch_mod(bi.model_name, bi.dataset)
        tmod = torchdig.DIGProtectedModule(tmod)

        wbbfa = WBBFA()

        train_loader = dataman.get_benign_loader(bi.dataset, bi.input_img_size, 'train', bi.batch_size)
        test_loader = dataman.get_sampling_loader_v2(
            bi.dataset, bi.input_img_size, 'test', bi.batch_size, n_per_class=bi.fast_n_per_class
        )

        # tmod.to('cuda')
        # WBBFA.get_sus_score_range(tmod, train_loader)
        # continue

        tmod_a = wbbfa.attack(
            tmod, train_loader, test_loader,
            sus_score_range=torchdig.cached_sus_score_ranges[(bi.model_name, bi.dataset)]
        )
