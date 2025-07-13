# Differentiable analytical placement with CG update and early-stop loop
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import scipy.ndimage
from scipy.ndimage import gaussian_filter
import numpy as np
import sys
import imageio
import math

class Placer(nn.Module):
    def __init__(self, chip_info, netlist, gamma=1.0, target_density=1.0, weight_increase_factor=1.5, trunc_factor=1.0, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.netlist = netlist
        self.chip_info = chip_info
        self.board_size = torch.tensor(chip_info.board_size, dtype=torch.float32, device=self.device)
        self.gamma = torch.tensor(gamma, device=self.device)
        self.num_bins = torch.tensor(chip_info.num_bins, dtype=torch.long, device=self.device)
        self.bin_size = torch.tensor(chip_info.bin_size, dtype=torch.float32, device=self.device)
        self.target_density = torch.tensor(chip_info.target_density, dtype=torch.float32, device=self.device)
        self.weight_increase_factor = torch.tensor(weight_increase_factor, dtype=torch.float32, device=self.device)
        self.trunc_factor = torch.tensor(trunc_factor, dtype=torch.float32, device=self.device)

        self.weight_density = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        # Cell parameters
        self.cells_pos = torch.tensor(netlist.cells_pos, dtype=torch.float32, device=self.device)
        self.is_movable = torch.tensor((1 - netlist.cells_isfixed).astype(int), dtype=torch.int, device=self.device).unsqueeze(1)
        self.cells_size = torch.tensor(netlist.cells_size, dtype=torch.float32, device=self.device)
        self.nets = [torch.tensor(net, dtype=torch.long, device=self.device) for net in netlist.nets]
        # torch.tensor(netlist.nets_pins_offset, dtype=torch.float32, device=self.device)
        self.nets_pins_offset = [torch.tensor(pins_offset, dtype=torch.float32, device=self.device) for pins_offset in netlist.nets_pins_offset]
        self.num_cells = len(netlist.cells_name)
        self.num_movable_cells = torch.sum(self.is_movable).item()

        # Bins
        self.bins_potential = torch.zeros((chip_info.num_bins, chip_info.num_bins), dtype=torch.float32, device=self.device)
        self.bins_free_space = np.full((chip_info.num_bins, chip_info.num_bins), chip_info.bin_size[0]*chip_info.bin_size[1])
        self.bins_used_area = np.zeros((chip_info.num_bins, chip_info.num_bins))
        self.total_movable_area = np.sum(np.prod(netlist.cells_size[netlist.cells_isfixed == 0], axis=1))
        self.bins_base_potential = torch.zeros((chip_info.num_bins, chip_info.num_bins), dtype=torch.float32, device=self.device)
        self.bins_expect_potential = torch.zeros((chip_info.num_bins, chip_info.num_bins), dtype=torch.float32, device=self.device)
        self.cells_bins_potential = {'cell_idx': None, 'bin_x': None, 'bin_y': None, 'potential': None}
        self.cells_potential_norm = torch.ones(self.num_cells, device=self.device)
        self.wDen = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # Visualize
        self.frames = []

        # ML
        self.learned_lambda_model = None
        self.MLdeterministic = False
    def set_eval_mode(self):
        self.MLdeterministic = True
    def set_train_mode(self):
        self.MLdeterministic = False

    def set_model(self, model):
        self.learned_lambda_model = model


    def set_pos(self, pos):
        if isinstance(pos, np.ndarray):
            self.cells_pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
            self.netlist.cells_pos = pos.copy()
        elif isinstance(pos, torch.Tensor):
            self.cells_pos = pos.clone()
            self.netlist.cells_pos = pos.detach().cpu().numpy()

    def bound_cells(self, pos):
        half_size = self.cells_size / 2
        pos = torch.max(pos, half_size)
        pos = torch.min(pos, self.board_size - half_size)
        return pos
    
    def initialize_placement(self): # np
        pos = np.random.rand(self.num_cells, 2) * self.chip_info.board_size  # Random positions within the canvas
        # random place movable cells at centor area (9/20 ~ 11/20)
        for i in range(self.num_cells):
            # rand float in [0.45, 0.55] * board_size
            if self.netlist.cells_isfixed[i] == 0:
                pos[i, 0] = (0.45 + 0.1 * np.random.rand()) * self.board_size[0]  # x
                pos[i, 1] = (0.45 + 0.1 * np.random.rand()) * self.board_size[1]  # y
        pos_tensor = torch.tensor(pos, dtype=torch.float32, device=self.device)
        pos_tensor = self.bound_cells(pos_tensor)
        pos = pos_tensor.detach().cpu().numpy()
        return pos
    def initialize_placement_with_fixed_fixed(self, num_cells, init_pos): # np
        pos = init_pos.copy()
        # random place movable cells at centor area (9/20 ~ 11/20)
        for i in range(num_cells):
            # rand float in [0.45, 0.55] * board_size
            if self.netlist.cells_isfixed[i] == 0:
                pos[i, 0] = (0.45 + 0.1 * np.random.rand()) * self.chip_info.board_size[0]  # x
                pos[i, 1] = (0.45 + 0.1 * np.random.rand()) * self.chip_info.board_size[1]  # y
        pos_tensor = torch.tensor(pos, dtype=torch.float32, device=self.device)
        pos_tensor = self.bound_cells(pos_tensor)
        pos = pos_tensor.detach().cpu().numpy()
        return pos

    # Wirelength Model
    def calculate_lse_wl(self):
        total_lse = torch.tensor(0.0, device=self.device)
        for net_idx, net in enumerate(self.nets):
            xs = self.cells_pos[net, 0] + self.nets_pins_offset[net_idx][:, 0]
            ys = self.cells_pos[net, 1] + self.nets_pins_offset[net_idx][:, 1]
            total_lse += self.gamma * (torch.logsumexp(xs / self.gamma, dim=0) + torch.logsumexp(-xs / self.gamma, dim=0))
            total_lse += self.gamma * (torch.logsumexp(ys / self.gamma, dim=0) + torch.logsumexp(-ys / self.gamma, dim=0))
        return total_lse
    
    def calculate_wl_gradient(self):
        grads = torch.zeros_like(self.cells_pos)
        for net_idx, net in enumerate(self.nets):
            # centor xy
            xs = self.cells_pos[net, 0] + self.nets_pins_offset[net_idx][:, 0]
            ys = self.cells_pos[net, 1] + self.nets_pins_offset[net_idx][:, 1] 
            # exponential elements
            exp_x = torch.exp(xs / self.gamma)       # max(x)
            exp_neg_x = torch.exp(-xs / self.gamma)  # -min(x)
            exp_y = torch.exp(ys / self.gamma)       # max(y)
            exp_neg_y = torch.exp(-ys / self.gamma)  # -min(y)
            # gradients (partial derivative for each x_i)
            grad_x = self.gamma * ( (exp_x / exp_x.sum()) - (exp_neg_x / exp_neg_x.sum()) )
            grad_y = self.gamma * ( (exp_y / exp_y.sum()) - (exp_neg_y / exp_neg_y.sum()) )

            grads[net, 0] += grad_x
            grads[net, 1] += grad_y
        return grads
    
    def calculate_hpwl(self): # np
        # HPWL_x = max(x) - min(x)
        hpwl = 0
        for net_idx, net in enumerate(self.nets): # for each net
            x_coords = self.netlist.cells_pos[net, 0] + self.netlist.nets_pins_offset[net_idx][:, 0]
            y_coords = self.netlist.cells_pos[net, 1] + self.netlist.nets_pins_offset[net_idx][:, 1]
            hpwl += (x_coords.max() - x_coords.min()) + (y_coords.max() - y_coords.min())
        return hpwl
    
    # Density Model
    def init_density_model(self, show_info=False):
        bins_base_potential = self.update_potential_base()
        bins_base_potential = self.smooth_base_potential(bins_base_potential)
        bins_expect_potential = self.update_exp_bin_potential(bins_base_potential, show_info=show_info)
        self.bins_base_potential = torch.tensor(bins_base_potential, dtype=torch.float32, device=self.device)
        self.bins_expect_potential = torch.tensor(bins_expect_potential, dtype=torch.float32, device=self.device)
        self.bins_potential = torch.tensor(bins_base_potential, dtype=torch.float32, device=self.device)

    def calculate_overflow_density(self): # np
        bin_size = self.chip_info.bin_size
        overflow = np.maximum(self.bins_used_area - self.bins_free_space, 0)
        # total overflow density
        total_over_den = np.sum(overflow) / self.total_movable_area + 1.0
        # max density (for all non-zero space bins)
        prepalced_area = bin_size[0]*bin_size[1] - self.bins_free_space
        density = (self.bins_used_area + prepalced_area) / (bin_size[0]*bin_size[1])
        max_den = np.max(density[self.bins_free_space > 0])
        return total_over_den, overflow, max_den
        
    def calculate_overflow_potential(self):
        #torch.pow(self.bins_potential - self.bins_expect_potential, 2)
        overflow = torch.maximum(self.bins_potential - self.bins_expect_potential, torch.tensor(0.0, device=self.device))
        total_overflow = overflow.sum()
        max_overflow = overflow.max()
        return total_overflow, overflow, max_overflow
    
    def calculate_density_gradient(self):
        device = self.device; dtype = self.cells_pos.dtype
        grads = torch.zeros_like(self.cells_pos)
        cell_idx = self.cells_bins_potential['cell_idx']
        bin_x = self.cells_bins_potential['bin_x']
        bin_y = self.cells_bins_potential['bin_y']
        potential = self.cells_bins_potential['potential']
        # Step 1: extend cell-bin pair as tensor
        x = self.cells_pos[cell_idx, 0]; y = self.cells_pos[cell_idx, 1]; w = self.cells_size[cell_idx, 0]; h = self.cells_size[cell_idx, 1]
        bin_cx = bin_x * self.bin_size[0] + self.bin_size[0] / 2; bin_cy = bin_y * self.bin_size[1] + self.bin_size[1] / 2
        # calculate gradient
        gx = self.get_potential_gradient(x, bin_cx, w, self.bin_size[0])
        gy = self.get_potential_gradient(y, bin_cy, h, self.bin_size[1])
        diff = self.bins_potential[bin_x, bin_y] - self.bins_expect_potential[bin_x, bin_y] # loss (distance between target)
        # each bin contribute to cell
        grad_x_contrib = gx * diff
        grad_y_contrib = gy * diff
        # Step 2: 根據 cell_idx 聚合
        grads_x = torch.zeros(self.num_cells, device=device, dtype=dtype)
        grads_y = torch.zeros(self.num_cells, device=device, dtype=dtype)
        grads_x = grads_x.index_add(0, cell_idx, grad_x_contrib)
        grads_y = grads_y.index_add(0, cell_idx, grad_y_contrib)
        
        grads = torch.stack([grads_x, grads_y], dim=1)
        return grads

    def get_potential(self, xv, xb, cell_size, bin_size):
        # Bell-Shaped potential: potential_x(b, v) --- Refer to Eq. (6,7) and Fig. 2(b) in the paper
        # Return the potential value of the overlap area (1-dir overlapping for one cell to one bin)
        d = torch.abs(xv - xb) if isinstance(xv, torch.Tensor) else abs(xv - xb)
        w = cell_size
        wb = bin_size
        if d <= w/2 + wb:
            a = 4 / ((w + 2*wb)*(w + 4*wb))
            return 1 - a * d**2
        elif d <= w/2 + 2*wb:
            b = 2 / (wb*(w + 4*wb))
            return b * (d - w/2 - 2*wb)**2
        else:
            return 0.0
    def get_potential_gradient(self, xv, xb, cell_size, bin_size): # To torch: support batch and autograd
        # partial derivative of potential_x(b, v) w.r.t. xv
        d = torch.abs(xv - xb)
        w = cell_size
        wb = bin_size
        # condiction (bell-shape)
        cond1 = d <= (w / 2 + wb) # a part
        cond2 = (d > (w / 2 + wb)) & (d <= (w / 2 + 2 * wb)) # b part
        cond3 = d > (w / 2 + 2 * wb) # grad=0
        # coeff
        a = 4 / ((w + 2 * wb) * (w + 4 * wb))
        b = 2 / (wb * (w + 4 * wb))
        sign = torch.where(xv >= xb, -1.0, 1.0)
        # grad
        grad = torch.zeros_like(xv)
        grad = torch.where(cond1, sign * 2 * a * d, grad)
        grad = torch.where(cond2, -sign * 2 * b * (d - w / 2 - 2 * wb), grad)
        return grad
    
    def get_overlap(self, xv, xb, cell_size, bin_size): # np
        # Return the overlap dist of one cell to one bin
        x1 = xv - cell_size / 2
        x2 = xv + cell_size / 2
        b1 = xb - bin_size / 2
        b2 = xb + bin_size / 2
        overlap = max(min(x2, b2) - max(x1, b1), 0)
        return overlap

    def find_contribute_bins(self, cx, cy, w, h): # np
        bin_size = self.chip_info.bin_size; num_bins = self.chip_info.num_bins
        # Find contributed bins (influence bin range < 2 bins around cell, bin_x = cell_cx +- cell_w/2 + 2*num_bins)
        min_bin_x = max(int((cx - (w/2 + 2*bin_size[0])) // bin_size[0]), 0)
        max_bin_x = min(int((cx + (w/2 + 2*bin_size[0])) // bin_size[0]), num_bins-1)
        min_bin_y = max(int((cy - (h/2 + 2*bin_size[1])) // bin_size[1]), 0)
        max_bin_y = min(int((cy + (h/2 + 2*bin_size[1])) // bin_size[1]), num_bins-1)
        return min_bin_x, max_bin_x, min_bin_y, max_bin_y

    def update_potential_base(self): # np
        bin_size = self.chip_info.bin_size; num_bins = self.chip_info.num_bins
        self.bins_free_space = np.full((num_bins, num_bins), bin_size[0]*bin_size[1]) # initial free space = bin_size*bin_size (empty bin)
        bins_base_potential = np.zeros((num_bins, num_bins)) # initial potential = 0.0
        for idx, cell_pos in enumerate(self.netlist.cells_pos):
            if self.netlist.cells_isfixed[idx] == 0: continue
            # skip movable cells, only for fixed cell
            cx, cy = cell_pos
            w, h = self.netlist.cells_size[idx]
            min_bin_x, max_bin_x, min_bin_y, max_bin_y = self.find_contribute_bins(cx, cy, w, h)
            # bx, by are bin index (not position)
            for bx in range(min_bin_x, max_bin_x + 1):
                for by in range(min_bin_y, max_bin_y + 1):
                    bin_cx = bx * bin_size[0] + bin_size[0] / 2
                    bin_cy = by * bin_size[1] + bin_size[1] / 2
                    # calculate potential
                    pot_x = self.get_potential(cx, bin_cx, w, bin_size[0])
                    pot_y = self.get_potential(cy, bin_cy, w, bin_size[1])
                    potential = pot_x * pot_y * w * h
                    bins_base_potential[bx, by] += potential
                    # calculate free space
                    overlap_x = self.get_overlap(cx, bin_cx, w, bin_size[0])
                    overlap_y = self.get_overlap(cy, bin_cy, w, bin_size[1])
                    overlap = overlap_x * overlap_y
                    self.bins_free_space[bx, by] = max(self.bins_free_space[bx, by] - overlap, 0)
        return bins_base_potential

    def smooth_base_potential(self, bins_base_potential, smooth_r=1.0, smooth_delta=1.0): # np # SmoothBasePotential() (smooth_r=5, smooth_delta=1.0)
        # Step 1: Gaussian smoothing
        smoothed = gaussian_filter(bins_base_potential, sigma=smooth_r)
        bins_base_potential[:, :] = smoothed
        # Step 2: Level smoothing (only for multi-level, smooth_delta > 1)
        if smooth_delta > 1.001:
            bins_base_potential = self.level_smooth_base_potential(bins_base_potential, smooth_delta) # pass by reference
            return bins_base_potential
        # Step 3: Additional height boost for fully blocked bins (Find the bins that are fully blocked but being too much smoothed by Gaussian)
        more_smooth = gaussian_filter(bins_base_potential, sigma=smooth_r * 6) # more smoothed version for boost
        bin_area = self.chip_info.bin_size[0] * self.chip_info.bin_size[1]
        half_bin_area = bin_area / 2
        scale = 3
        for i in range(bins_base_potential.shape[0]):
            for j in range(bins_base_potential.shape[1]):
                free = bin_area - bins_base_potential[i][j]
                # If the bin's potential is already no space or high enough, boost its potential
                if free < 1e-4 and more_smooth[i][j] > half_bin_area:
                    bins_base_potential[i][j] += (more_smooth[i][j] - half_bin_area) * scale
        return bins_base_potential
    def level_smooth_base_potential(self, basePotential, delta): # np # LevelSmoothBasePotential
        oldPotential = basePotential.copy()
        maxPotential = np.max(oldPotential)
        totalPotential = np.sum(oldPotential)
        avgPotential = totalPotential / oldPotential.size
        if totalPotential == 0:
            return  # 無 preplaced block
        # Apply TSP-style smoothing
        newPotential = np.zeros_like(oldPotential)
        for i in range(basePotential.shape[0]):
            for j in range(basePotential.shape[1]):
                val = oldPotential[i][j]
                if val >= avgPotential:
                    newPotential[i][j] = avgPotential + ((val - avgPotential) / maxPotential) ** delta * maxPotential
                else:
                    newPotential[i][j] = avgPotential - ((avgPotential - val) / maxPotential) ** delta * maxPotential
        # Normalize total potential to match original
        newTotal = np.sum(newPotential)
        ratio = totalPotential / newTotal if newTotal != 0 else 1.0
        basePotential[:, :] = newPotential * ratio
        return basePotential

    def update_exp_bin_potential(self, bins_base_potential, target_util=1.0, show_info=False): # np # UpdateExpBinPotential
        bin_size = self.chip_info.bin_size; num_bins = self.chip_info.num_bins
        bins_expect_potential = np.zeros_like(bins_base_potential)
        total_free = 0
        zero_bin = 0
        for i in range(num_bins):
            for j in range(num_bins):
                base = bins_base_potential[i, j]
                # Calculate the free space available in the bin
                free = bin_size[0] * bin_size[1] - base  # Simplified for illustration
                if( free > 1e-4 ):
                    bins_expect_potential[i, j] = free * target_util
                    total_free += bins_expect_potential[i, j]
                else:
                    zero_bin += 1
        
        total_movable_module_area = np.sum(self.netlist.cells_size[self.netlist.cells_isfixed == 0][:, 0] * self.netlist.cells_size[self.netlist.cells_isfixed == 0][:, 1])
        alg_util = total_movable_module_area / total_free
        if show_info:
            print(f"PBIN: Zero space bin #= {zero_bin} ({100*zero_bin/num_bins/num_bins}%).  Algorithm utilization= {alg_util}")
        return bins_expect_potential

    def compute_new_potential_grid(self):
        num_cells = self.num_cells
        cell_idx_list = []
        bin_x_list = []
        bin_y_list = []
        potential_list = []
        self.cells_potential_norm = torch.ones(num_cells, device=self.device)
        for i in range(self.num_cells):
            if self.netlist.cells_isfixed[i]: continue
            # skip fixed cells
            x, y = self.cells_pos[i]
            w, h = self.cells_size[i]
            total_potential = 0.0
            # Find contributed bins (influence bin range < 2 bins around cell, bin_x = cell_cx +- cell_w/2 + 2*num_bins)
            min_bin_x, max_bin_x, min_bin_y, max_bin_y = self.find_contribute_bins(self.netlist.cells_pos[i,0], self.netlist.cells_pos[i,1], self.netlist.cells_size[i,0], self.netlist.cells_size[i,1])
            # bx, by are bin index (not position)
            for bx in range(min_bin_x, max_bin_x + 1):
                for by in range(min_bin_y, max_bin_y + 1):
                    bx_ts = torch.tensor(bx, device=self.device)
                    by_ts = torch.tensor(by, device=self.device)
                    bin_cx = bx_ts * self.bin_size[0] + self.bin_size[0] / 2
                    bin_cy = by_ts * self.bin_size[1] + self.bin_size[1] / 2
                    potential_x = self.get_potential(x, bin_cx, w, self.bin_size[0])
                    potential_y = self.get_potential(y, bin_cy, h, self.bin_size[1])
                    potential = potential_x * potential_y * w * h  # area contribution
                    if potential > 0:
                        cell_idx_list.append(i)
                        bin_x_list.append(bx)
                        bin_y_list.append(by)
                        potential_list.append(potential) # NOT .item()!
                        total_potential = total_potential + potential
            if total_potential > 0:
                cell_area = w * h
                self.cells_potential_norm[i] = cell_area / total_potential
        # Combine to tensor（autograd-safe）
        self.cells_bins_potential = { # Sparse coordinates
            'cell_idx': torch.tensor(cell_idx_list, device=self.device, dtype=torch.long),
            'bin_x': torch.tensor(bin_x_list, device=self.device, dtype=torch.long),
            'bin_y': torch.tensor(bin_y_list, device=self.device, dtype=torch.long),
            'potential': torch.stack(potential_list),  # float tensor
        }

    def update_potential_grid(self):
        dtype = self.cells_pos.dtype
        bin_size = self.chip_info.bin_size; num_bins = self.chip_info.num_bins
        self.bins_potential = torch.zeros((num_bins, num_bins), device=self.device, dtype=dtype)
        # Step 1: Update bin potential（scatter sum）
        cell_idx = self.cells_bins_potential['cell_idx']
        bin_x = self.cells_bins_potential['bin_x']
        bin_y = self.cells_bins_potential['bin_y']
        potential = self.cells_bins_potential['potential']
        normed_pot = potential * self.cells_potential_norm[cell_idx]
        # to 1d
        bin_flat_idx = bin_x * self.num_bins + bin_y
        bins_potential_flat = torch.zeros(num_bins * num_bins, device=self.device, dtype=dtype)
        bins_potential_flat = bins_potential_flat.index_add(0, bin_flat_idx, normed_pot)
        self.bins_potential = bins_potential_flat.view(num_bins, num_bins)
        # Step 2: Compute used area in each bin（For early stop）# np
        self.bins_used_area = np.zeros((self.num_bins, self.num_bins))
        cell_idx_np = cell_idx.detach().cpu().numpy(); bin_x_np = bin_x.detach().cpu().numpy(); bin_y_np = bin_y.detach().cpu().numpy()
        x1, y1 = self.netlist.cells_pos[:, 0] - self.netlist.cells_size[:, 0] / 2, self.netlist.cells_pos[:, 1] - self.netlist.cells_size[:, 1] / 2
        x2, y2 = self.netlist.cells_pos[:, 0] + self.netlist.cells_size[:, 0] / 2, self.netlist.cells_pos[:, 1] + self.netlist.cells_size[:, 1] / 2
        for cid, bx, by in zip(cell_idx_np, bin_x_np, bin_y_np):
            x1, x2, y1, y2 = self.netlist.cells_pos[cid][0] - self.netlist.cells_size[cid][0]/2, self.netlist.cells_pos[cid][0] + self.netlist.cells_size[cid][0]/2, self.netlist.cells_pos[cid][1] - self.netlist.cells_size[cid][1]/2, self.netlist.cells_pos[cid][1] + self.netlist.cells_size[cid][1]/2
            bx1, bx2, by1, by2 = bx * bin_size[0], (bx+1) * bin_size[0], by * bin_size[1], (by+1) * bin_size[1]
            overlap_area = max(0, min(x2, bx2) - max(x1, bx1)) * max(0, min(y2, by2) - max(y1, by1))
            self.bins_used_area[bx, by] += overlap_area
        return

    # Main Objective Function
    def calculate_obj_value(self, total_wl, total_potential, _lambda):
        obj_value = total_wl + 0.5 * _lambda * total_potential
        return obj_value
    
    def calculate_objective_gradient(self):
        grad_wl = self.calculate_wl_gradient()         # shape: (num_cells, 2)
        grad_pot = self.calculate_density_gradient()
        # Combine gradient（wirelength + density）
        grads = grad_wl + 0.5 * grad_pot * self.weight_density
        # Adjust force（gradient clipping）
        grads = self.adjust_force(grads)
        return grads, grad_wl, grad_pot

    def adjust_force(self, gradients):
        # 1. calculate gradient square and mean
        grad_sqr = gradients[:, 0]**2 + gradients[:, 1]**2
        avg_grad = torch.sqrt(torch.mean(grad_sqr))
        exp_max_grad = avg_grad * self.trunc_factor
        exp_max_grad_sqr = exp_max_grad**2
        # 2. construct clipping mask
        mask = grad_sqr > exp_max_grad_sqr
        scale = exp_max_grad / torch.sqrt(grad_sqr + 1e-12)
        # 3. use scale（only for gradients over expected
        adjusted = torch.where(mask.unsqueeze(1), gradients * scale.unsqueeze(1), gradients)
        return adjusted

    def init_weight_density(self):
        # Update potential
        self.compute_new_potential_grid()
        self.update_potential_grid()
        # Get gradients
        grad_wl = self.calculate_wl_gradient() 
        grad_pot = self.calculate_density_gradient()
        grad_wl_sum = torch.sum(torch.abs(grad_wl))
        grad_pot_sum = torch.sum(torch.abs(grad_pot))
        # Get inital density weight
        self.weight_density = grad_wl_sum / (grad_pot_sum + 1e-8)
        return self.weight_density
    def update_weight_density(self, iter):
        self.weight_density = self.weight_density * self.weight_increase_factor
        # self.weight_density = self.init_weight_density()
        # self.weight_density = self.weight_density * pow(2, iter*0.5)
        return self.weight_density
    def set_weight_density(self, new_weight_density):
        self.weight_density = torch.tensor(new_weight_density, dtype=torch.float32, device=self.device)


    # Main Funcitons
    def solve_placement(self, iter, step_size=1.0, min_inner_iter=0, max_inner_iter=50, bUseML=False, bPlot=False, bSaveFrames=False): # inner loop for solving subproblem (keep same density weight and solve until converge)
        inner_early_stop = False
        lambda_scale_list = []
        inner_cost_info_list = []
        total_lse = self.calculate_lse_wl()
        total_potential_overflow, potential_overflow_map, max_pot = self.calculate_overflow_potential()
        total_density_overflow, density_overflow_map, max_den = self.calculate_overflow_density()
        wl_pot_balance_ratio = total_lse / (total_potential_overflow+1e-1)
        first_total_obj_value = total_obj_value = self.calculate_obj_value(total_lse, total_potential_overflow, self.weight_density * wl_pot_balance_ratio)
        
        inner_iter = 0
        g_prev = grads = torch.zeros_like(self.cells_pos)  # init grad = 0
        d_prev = dircts = torch.zeros_like(self.cells_pos)  # init dir = 0
        inner_loss_list = []
        while(not inner_early_stop):
            # Update inner last values
            inner_obj_prev = total_obj_value
            pos_prev = self.netlist.cells_pos.copy()
            g_prev = grads; d_prev = dircts
            # Calculate gradient
            grad_wl = self.calculate_wl_gradient()         # shape: (num_cells, 2)
            grad_pot = self.calculate_density_gradient()
            # Combine gradient（wirelength + density）
            grad = None
            if bUseML:
                # prepare features
                features = self.get_features(self.cells_pos, self.cells_size, grads, g_prev, d_prev, grad_wl, grad_pot, self.weight_density, total_density_overflow, total_potential_overflow, max_den, self.is_movable)
                local_maps = self.get_local_maps(potential_overflow_map, density_overflow_map, step_size)
                # get lambda scale
                lambda_scale_movable = self.learned_lambda_model(features, local_maps) # (M, 2) from model
                # build full lambda scale tensor (N, 2), fixed cells have scale=1
                lambda_scale = torch.ones_like(self.cells_pos)
                mask = self.is_movable.squeeze().bool() # shape: (N,2)
                lambda_scale = lambda_scale.clone() 
                lambda_scale[mask] = lambda_scale_movable
                # new lambda for each cell
                lambda_new = self.weight_density * lambda_scale
                # if lambda_new
                # combine gradient
                grad = grad_wl + 0.5 * grad_pot * lambda_new
                # save lambda scale
                lambda_scale_list.append(lambda_scale_movable.detach().cpu().numpy())
            else:
                grad = grad_wl + 0.5 * grad_pot * self.weight_density
            # Adjust force（gradient clipping）
            grads = self.adjust_force(grad)
            # Update positions
            cells_pos, dircts = self.update_positions_cg(self.cells_pos, g_prev, d_prev, grads, step_size=step_size)
            cells_pos = self.bound_cells(cells_pos)
            self.set_pos(cells_pos)
            # Update density
            self.compute_new_potential_grid()
            self.update_potential_grid()

            # Calculate wirelength
            total_lse = self.calculate_lse_wl()
            total_hpwl = self.calculate_hpwl()
            # Calculate Density
            total_potential_overflow, potential_overflow_map, max_pot = self.calculate_overflow_potential()
            total_density_overflow, density_overflow_map, max_den = self.calculate_overflow_density()
            # Calculate objective value
            total_obj_value = self.calculate_obj_value(total_lse, total_potential_overflow, self.weight_density * wl_pot_balance_ratio)
            total_cost = self.calculate_obj_value(total_hpwl, total_density_overflow, self.weight_density.item())

            # Store loss
            # if bUseML:
            #     lambda_penalty_weight = 0.1
            #     lambda_penalty = ((lambda_scale_movable[:,0] - 1.0) ** 2 + (lambda_scale_movable[:,1] - 1.0) ** 2).mean()
            #     loss = -(first_total_obj_value-total_obj_value)/first_total_obj_value + lambda_penalty_weight * lambda_penalty
            # else:
            loss = -(first_total_obj_value-total_obj_value)/first_total_obj_value
            # loss = total_obj_value
            inner_loss_list.append(loss)

            # Plot placement
            if bPlot:
                fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), facecolor='white')
                costs_info = {"total_hpwl": total_hpwl, "total_lse": total_lse.item(), "total_density_overflow": total_density_overflow, "total_potential_overflow": total_potential_overflow.item(), "total_obj_value": total_obj_value.item(), "total_cost": total_cost, "weight_density": self.weight_density.item(), "gWL": np.sum(np.abs(grad_wl.detach().cpu().numpy())), "gPot": np.sum(np.abs(grad_pot.detach().cpu().numpy()))}
                density_map = self.bins_potential.detach().cpu().numpy() + self.bins_base_potential.detach().cpu().numpy()
                overflow_map = np.maximum(self.bins_potential.detach().cpu().numpy() - self.bins_expect_potential.detach().cpu().numpy(), 0)
                self.plot_placement(axs[0], iter, inner_iter, costs_info)
                self.plot_density_map(fig, axs[1], "Density Map", density_map, _vmax=2)
                self.plot_density_map(fig, axs[2], "Overflow Map", overflow_map, _vmax=2)
                display(fig)
                clear_output(wait=True)
                plt.close(fig)
                plt.pause(0.001)
                if bSaveFrames:
                    fig.canvas.draw()
                    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    self.frames.append(frame)

            # Check break inner loop
            inner_enough_iter = inner_iter > max_inner_iter
            inner_cannot_further_improve = total_obj_value.item() >= (inner_obj_prev * 0.99999)
            if (inner_cannot_further_improve) and (inner_iter >= min_inner_iter):
                # print(f"inner_cannot_further_improve: {inner_cannot_further_improve}")
                inner_early_stop = True
            obj_diff_too_small = abs(inner_obj_prev - total_obj_value.item()) < 1e-2
            grad_too_small = np.linalg.norm(grads.detach().cpu().numpy()) < 1e-2
            move_too_small = np.linalg.norm(abs(pos_prev-self.netlist.cells_pos)) < 1e-2
            if (inner_enough_iter or obj_diff_too_small or grad_too_small or move_too_small) and (inner_iter >= min_inner_iter):
                # print(f"obj_diff_too_small: {obj_diff_too_small}, grad_too_small: {grad_too_small}, move_too_small: {move_too_small}")
                inner_early_stop = True

            # end of one inner iter
            inner_iter += 1
            # cost_info_list
            cost_info = {"total_hpwl": total_hpwl, "total_lse": total_lse.item(), "total_density_overflow": total_density_overflow, "total_potential_overflow": total_potential_overflow.item(), "total_obj_value": total_obj_value.item(), "total_cost": total_cost, "weight_density": self.weight_density.item(), "gWL": np.sum(np.abs(grad_wl.detach().cpu().numpy())), "gPot": np.sum(np.abs(grad_pot.detach().cpu().numpy())), "max_den": max_den, "wl_pot_balance_ratio": wl_pot_balance_ratio.item()}
            inner_cost_info_list.append(cost_info)
        return inner_cost_info_list, inner_loss_list, lambda_scale_list

        
    def update_positions_cg(self, cells_pos: torch.Tensor, g_prev: torch.Tensor, d_prev: torch.Tensor, grads: torch.Tensor, step_size: float = 1.0):
        """
        Conjugate Gradient update for differentiable tensor-based placement.
        Args:
            cells_pos: (N, 2) current cell positions (requires_grad=True)
            g_prev: (N, 2) previous gradient
            d_prev: (N, 2) previous direction
            grad:   (N, 2) current gradient
            step_size: scalar step size (float)
            is_fixed: (N,) boolean tensor indicating fixed cells

        Returns:
            new_pos: (N, 2) updated positions (with gradient)
            new_dir: (N, 2) updated direction (for next iter)
        """
        # (1) We have gradient directions grad = g_k = ∇f(x_k)
        # (2) Compute Polak-Ribiere parameter β_k
        beta = self.compute_beta_polak_ribiere_xy(grads, g_prev)
        # (3) Compute conjugate directions d = -grad + beta*d_prev
        direct = -grads + beta * d_prev
        # (4) Compute step size \alpha = s/norm(d)
        alpha = step_size / (direct.view(-1).norm() + 1e-8)
        # (5) update positions: x = x_prev + alpha*d
        new_pos = cells_pos + alpha * direct * self.is_movable

        return new_pos, direct
    
    def compute_beta_polak_ribiere_xy(self, grads: torch.Tensor, g_prev: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        grad, g_prev: shape (N, 2)
        returns: beta: a value
        """
        # Make grad and g_prev flatten
        grads_flat = grads.view(-1)
        g_prev_flat = g_prev.view(-1)
        # Compute maximum absolute value for scaling to avoid overflow
        max_grad = torch.maximum(grads_flat.abs().max(), g_prev_flat.abs().max())
        max_grad = torch.clamp(max_grad, min=eps)
        if max_grad < 1e-10:
            return torch.tensor(0)
        # Normalize the gradients
        grad_scaled = grads_flat / max_grad
        g_prev_scaled = g_prev_flat / max_grad
        # Numerator: g_k^T (g_k - g_{k-1})
        diff_scaled = (grads_flat - g_prev_flat) / max_grad
        numerator = torch.dot(grad_scaled, diff_scaled) # product
        # Denominator: sum(g_{k-1}^2)
        denominator = torch.clamp(g_prev_scaled.pow(2).sum(), min=eps) # l2 norm, avoid divide by 0
        # Beta: Polak-Ribiere formula
        beta = numerator / denominator
        return beta
    
    # Visualization
    def plot_placement(self, ax, iteration, inner_iter, costs_info, draw_net=True, show_name=True):
        bin_size = self.chip_info.bin_size; board_size = self.chip_info.board_size; num_bins = self.chip_info.num_bins
        for i in range(num_bins + 1):
            x = i * bin_size[0]
            y = i * bin_size[1]
            ax.axhline(y, color='lightgray', linewidth=0.5, zorder=0)
            ax.axvline(x, color='lightgray', linewidth=0.5, zorder=0)

        if draw_net:
            for net_idx, net in enumerate(self.nets):
                xs = self.netlist.cells_pos[net, 0] + self.netlist.nets_pins_offset[net_idx][:, 0]
                ys = self.netlist.cells_pos[net, 1] + self.netlist.nets_pins_offset[net_idx][:, 1]
                ax.plot(xs, ys, 'r--', alpha=0.5)  # Draw nets

        for i, (pos, size) in enumerate(zip(self.netlist.cells_pos, self.netlist.cells_size)):
            rect = plt.Rectangle(pos - size / 2, size[0], size[1], edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            if show_name:
                ax.text(pos[0], pos[1], self.netlist.cells_name[i], ha='center', va='center', fontsize=8, color='blue')

        ax.set_xlim(0, board_size[0])
        ax.set_ylim(0, board_size[1])
        # ax.set_title(f'Placement at Iteration {iteration}, WL(LSE/HPWL): {costs_info["total_lse"]:.2f} / {costs_info["total_hpwl"]:.2f}, Overflow(Pot/Den): {costs_info["total_potential_overflow"]:.2f} / {costs_info["total_density_overflow"]:.2f}, Cost(obj/real): {costs_info["total_obj_value"]:.2f} / {costs_info["total_cost"]:.2f}')
        ax.set_title(f'  WL: {costs_info["total_lse"]:.2f}, Overflow: {costs_info["total_potential_overflow"]:.2f},    Obj_f: {costs_info["total_obj_value"]:.2f}\n' + 
                    # f'  Grad: {costs_info["gWL"]:.2e} + {costs_info["weight_density"]:.2e} * {costs_info["gPot"]:.2e} (Den Ratio: {(costs_info["weight_density"]*costs_info["gPot"]/costs_info["gWL"]):.2e})\n' + 
                    f'HPWL: {costs_info["total_hpwl"]:.2f},  OverDen: {costs_info["total_density_overflow"]:.2f}, RealCost: {costs_info["total_cost"]:.2f}\n' + 
                    f'Placement at Iteration {iteration} ({inner_iter})')
        ax.grid(False)
    def plot_density_map(self, fig, ax, title, density_map, _vmax=None):
        n_bins = (density_map.shape[0], density_map.shape[1])
        # Draw heat map
        if(_vmax==None):
            heatmap = ax.imshow(
                density_map.T,              # transpose: make y-axis as vertical
                origin='lower',             # let (0, 0) be at the bottom-left
                extent=[0, n_bins[0], 0, n_bins[1]],  # extend to chip size
                cmap='Reds',                # heat map
                aspect='equal',
                vmin=0
            )
        else:
            heatmap = ax.imshow(
                density_map.T,              # transpose: make y-axis as vertical
                origin='lower',             # let (0, 0) be at the bottom-left
                extent=[0, n_bins[0], 0, n_bins[1]],  # extend to chip size
                cmap='Reds',                # heat map
                aspect='equal',
                vmin=0,
                vmax=_vmax
            )
        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.set_label('Density')
        for x in range(n_bins[0] + 1): ax.axvline(x, color='lightgray', linewidth=0.5)
        for y in range(n_bins[1] + 1): ax.axhline(y, color='lightgray', linewidth=0.5)
        ax.set_xlim(0, n_bins[0])
        ax.set_ylim(0, n_bins[1])
        ax.set_title(title)
        ax.grid(False)
    
    def plot_local_map(loc_map):
        n_bins = (loc_map.shape[0], loc_map.shape[1])
        fig, ax = plt.subplots()
        heatmap = ax.imshow(loc_map.T, origin='lower', cmap='Reds', extent=[0, n_bins[0], 0, n_bins[1]], vmax=1)
        for x in range(n_bins[0] + 1): ax.axvline(x, color='lightgray', linewidth=0.5)
        for y in range(n_bins[1] + 1): ax.axhline(y, color='lightgray', linewidth=0.5)
        plt.colorbar(heatmap, ax=ax)
        plt.show()
    
    def save_gif(self, gif_file_name):
        final_frames = self.frames.copy()
        for _ in range(2*5): # 2s * fps=5 (make it stop for 2s)
            final_frames.append(self.frames[-1])
        imageio.mimsave(gif_file_name, final_frames, fps=5, loop=0)
    def reset_frames(self):
        self.frames = []

    ################################## Training ##############################
    # Perparameter optimizer for each cell
    def get_features(self, cells_pos, cells_size, grad, g_prev, d_prev, gWL, gPot, weight_density, total_density_overflow, total_potential_overflow, max_den, is_movable):
        """
        Compose 22-dimensional features for movable cells only.
        Returns: (M, 22) tensor
        """
        N = cells_pos.size(0)
        wd = weight_density.to(self.device).expand(N, 1) # weight_density.ndim == 0 else weight_density.unsqueeze(1)
        od = torch.tensor(total_density_overflow, dtype=torch.float32).to(self.device).expand(N, 1)
        op = total_potential_overflow.to(self.device).expand(N, 1)
        md = torch.tensor(max_den, dtype=torch.float32).to(self.device).expand(N, 1)
        # calculate average gWL and gPot
        gWL_avg = torch.mean(gWL).to(self.device).expand(N, 1)
        gPot_avg = torch.mean(gPot).to(self.device).expand(N, 1)
        gWL_std = torch.std(gWL).to(self.device).expand(N, 1)
        gPot_std = torch.std(gPot).to(self.device).expand(N, 1)

        full_features = torch.cat([cells_pos, cells_size, grad, g_prev, d_prev, gWL, gPot, wd, od, op, md, gWL_avg, gPot_avg, gWL_std, gPot_std], dim=1)
        movable_mask = is_movable.squeeze().bool()  # shape: (N,)

        return full_features[movable_mask]  # shape: (M, 22)

    def get_local_maps(self, potential_overflow_map, density_overflow_map, step_size):
        # map: (2, h, w): [potential_overflow_map, density_overflow_map]
        # (0) Parameters for local view
        view_channel = 2
        cell_region = 3 # 3*3 center bins
        den_model_padding = 2 # 2 padding bins
        k_step_padding = 3 # 3 *step_size padding bins
        total_padding = den_model_padding + math.ceil(k_step_padding*step_size)
        local_view_w = local_view_h = cell_region + 2*total_padding
        # (1) Prepare location map for each cell
        num_bins = self.chip_info.num_bins; N = len(self.netlist.cells_pos)
        movable_indices = torch.nonzero(self.is_movable.squeeze()).squeeze()

        loc_views = torch.zeros((len(movable_indices), view_channel, local_view_w, local_view_h), dtype=torch.float32, device=self.device)  # (N, 1, W, H)
        density_overflow_map_ts = torch.from_numpy(density_overflow_map).to(potential_overflow_map.device)
        maps = torch.stack([potential_overflow_map, density_overflow_map_ts], dim=0)  # shape: (2, H, W)
        for i, cidx in enumerate(movable_indices):
            for j in range(view_channel):
                base_map = maps[j]
                padded_global_map = F.pad(base_map, (total_padding, total_padding, total_padding, total_padding), "constant", 10)
                ## (1.1) Get cell region with bin indexs
                cx, cy = self.netlist.cells_pos[cidx]; w, h = self.netlist.cells_size[cidx]; bin_size = self.chip_info.bin_size
                bx1, bx2, by1, by2 = math.floor((cx-w/2) / bin_size[0]), math.ceil((cx+w/2) / bin_size[0]), math.floor((cy-h/2) / bin_size[1]), math.ceil((cy+h/2) / bin_size[1])
                ## (1.2) Extract cell region and downsample to 3x3 with interpolate
                loc_cell = base_map[bx1:bx2, by1:by2]
                loc_cell_unsqueezed = loc_cell.unsqueeze(0).unsqueeze(0)
                loc_center = F.interpolate(loc_cell_unsqueezed, size=(cell_region, cell_region), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                ## (1.3) Extract local padding view with 2*3*step_size bins (6s+cw * 6s+ch)
                padding_bx1, padding_bx2, padding_by1, padding_by2 = bx1+total_padding, bx2+total_padding, by1+total_padding, by2+total_padding
                padding_crop_bx1, padding_crop_bx2, padding_crop_by1, padding_crop_by2 = padding_bx1-total_padding, padding_bx2+total_padding, padding_by1-total_padding, padding_by2+total_padding
                #|-----------------------|
                #|         Top           |
                #|-----------------------|
                #| Left | Center | Right |
                #|-----------------------|
                #|        Bottom         |
                #------------------------|
                ### Top
                top_crop = padded_global_map[padding_crop_bx1:padding_crop_bx2, padding_by2:padding_crop_by2]
                top_crop_unsqueezed = top_crop.unsqueeze(0).unsqueeze(0)
                loc_top = F.interpolate(top_crop_unsqueezed, size=(local_view_w, total_padding), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                ### Bottom
                bottom_crop = padded_global_map[padding_crop_bx1:padding_crop_bx2, padding_crop_by1:padding_by1]
                bottom_crop_unsqueezed = bottom_crop.unsqueeze(0).unsqueeze(0)
                loc_bottom = F.interpolate(bottom_crop_unsqueezed, size=(local_view_w, total_padding), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                ### Left
                left_crop = padded_global_map[padding_crop_bx1:padding_bx1, padding_by1:padding_by2]
                left_crop_unsqueezed = left_crop.unsqueeze(0).unsqueeze(0)
                loc_left = F.interpolate(left_crop_unsqueezed, size=(total_padding, cell_region), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                ### Right
                right_crop = padded_global_map[padding_bx2:padding_crop_bx2, padding_by1:padding_by2]
                right_crop_unsqueezed = right_crop.unsqueeze(0).unsqueeze(0)
                loc_right = F.interpolate(right_crop_unsqueezed, size=(total_padding, cell_region), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                ## (1.4) Concat all local views
                middle_row = torch.cat([loc_left, loc_center, loc_right], dim=0)
                loc_views[i][j] = torch.cat([loc_bottom, middle_row, loc_top], dim=1)

        return loc_views  # shape: (M, 2, W, H)
