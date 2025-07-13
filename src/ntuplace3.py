import sys, os
import numpy as np
import json
import matplotlib.pyplot as plt

from parser_bookshelf import Parser
from placedb import ChipInfo
from placer import Placer
from util import bcolors, ArgxParser
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^IMPORT^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^##
def parse_cfg():
    cfg = {}
    argxParser = ArgxParser()
    args = argxParser.parser.parse_args()
    with open(args.cfg) as json_f:
        cfg = json.load(json_f)
        cfg['cfg_file'] = args.cfg
    print("Overwrite configs with given flags:")
    for arg in vars(args):
        if getattr(args, arg) != None:
            cfg[arg] = getattr(args, arg)
            print(f"  {arg} = {getattr(args, arg)}")
    return cfg

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^FUNCTION^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^##


## Main
def main():
    ## Preliminary Information ##
    print("     #############################################################")
    print("     #                                                           #")
    print("     #                         [EDA LAB]                         #")
    print("     #-----------------------------------------------------------#")
    print("     #                        NTUPlace3ml                        #")
    print("     #                                                           #")
    print("     #############################################################")
    print("")

    ## Parse config file and args
    cfg = parse_cfg()

    ## Parse bookshelf file
    parser = Parser(cfg['aux'])
    parser.parse_bookshelf(print_info=False)

    ## Construct ChipInfo
    chip_info = ChipInfo(cfg)
    chip_info.set_from_parser(parser)
    chip_info.print_info()

    ## Init Placer
    placer = Placer(chip_info, chip_info.netlist, gamma=cfg['GAMMA'], target_density=cfg['target_density'], weight_increase_factor=cfg['WeightIncreaseFactor'], trunc_factor=cfg['TRUNC_FACTOR'])
    print("Initialize Placer!")
    np.random.seed(cfg['seed'])  # For reproducibilit
    init_pos = placer.initialize_placement() if chip_info.netlist.cells_pos.all() == 0 else chip_info.netlist.cells_pos.copy()
    
    import time
    # Initialize bins expected potential
    start = time.time()
    placer.init_density_model(show_info=True)
    end = time.time()
    print(f"Init_density_model, Time taken: {end - start:.3f} seconds")
    # Visualization Check 
    start = time.time()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    costs_info = {"total_hpwl": 0, "total_lse": 0, "total_density_overflow": 0, "total_potential_overflow": 0, "total_obj_value": 0, "total_cost": 0, "weight_density": 0, "gWL": 1, "gPot": 0}
    density_map = np.maximum(placer.bins_potential.detach().cpu().numpy() - placer.bins_expect_potential.detach().cpu().numpy(), 0)
    placer.plot_placement(axs[0], 0, 0, costs_info, draw_net=False, show_name=False)
    placer.plot_density_map(fig, axs[1], "Base Density Map", density_map, _vmax=2)
    plt.savefig("base.png")
    end = time.time()
    print(f"Plot_placement, Time taken: {end - start:.3f} seconds")
    # Initialize placement
    start = time.time()
    weight_density = placer.init_weight_density()
    end = time.time()
    print(f"Init_weight_density, Time taken: {end - start:.3f} seconds")
    # Visualization Check 
    start = time.time()
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), facecolor='white')
    costs_info = {"total_hpwl": 0, "total_lse": 0, "total_density_overflow": 0, "total_potential_overflow": 0, "total_obj_value": 0, "total_cost": 0, "weight_density": 0, "gWL": 1, "gPot": 0}
    density_map = placer.bins_potential.detach().cpu().numpy() + placer.bins_base_potential.detach().cpu().numpy()
    overflow_map = np.maximum(placer.bins_potential.detach().cpu().numpy() - placer.bins_expect_potential.detach().cpu().numpy(), 0)
    placer.plot_placement(axs[0], 0, 0, costs_info, draw_net=False, show_name=False)
    placer.plot_density_map(fig, axs[1], "Density Map", density_map, _vmax=2)
    placer.plot_density_map(fig, axs[2], "Overflow Map", overflow_map, _vmax=2)
    plt.savefig("init.png")
    end = time.time()
    print(f"Plot_placement, Time taken: {end - start:.3f} seconds")

    # Analytical Placement
    start = time.time()
    step_size = min(chip_info.board_size) / chip_info.num_bins
    cg_loss_list = []
    cost_info_list = []
    bPlot = False
    ## init
    placer.reset_frames()
    early_stop = False
    not_efficient = False
    placer.set_pos(init_pos)
    weight_density = placer.init_weight_density()
    total_lse = placer.calculate_lse_wl()
    total_hpwl = placer.calculate_hpwl()
    total_density_overflow, _, max_den = placer.calculate_overflow_density()
    total_potential_overflow, _, _ = placer.calculate_overflow_potential()
    grad_wl = placer.calculate_wl_gradient(); grad_pot = placer.calculate_density_gradient()
    wl_pot_balance_ratio = total_lse / total_potential_overflow
    total_obj_value = placer.calculate_obj_value(total_lse, total_potential_overflow, weight_density * wl_pot_balance_ratio)
    cost_info = {"total_hpwl": total_hpwl, "total_lse": total_lse.item(), "total_density_overflow": total_density_overflow, "total_potential_overflow": total_potential_overflow.item(), "total_obj_value": total_obj_value.item(), "weight_density": weight_density.item(), "gWL": np.sum(np.abs(grad_wl.detach().cpu().numpy())), "gPot": np.sum(np.abs(grad_pot.detach().cpu().numpy())), "max_den": max_den, "wl_pot_balance_ratio": wl_pot_balance_ratio.item()}
    cost_info_list.append([cost_info])
    ## run
    for iter in range(cfg['MAX_ITER']):
        # Update last values
        total_over_den_prev = total_density_overflow
        max_den_prev = max_den
        ############################## Solve sub placement problem with current density weight ##############################################
        cost_info, losses, _ = placer.solve_placement(iter, step_size=step_size, min_inner_iter=cfg['min_inner_iter'], max_inner_iter=cfg['max_inner_iter'], bPlot=bPlot, bSaveFrames=True)
        cg_loss_list.extend(losses)
        cost_info_list.append(cost_info)
        total_density_overflow = cost_info[-1]["total_density_overflow"]; total_lse = cost_info[-1]["total_lse"]; total_potential_overflow = cost_info[-1]["total_potential_overflow"]; total_obj_value = cost_info[-1]["total_obj_value"]; max_den = cost_info[-1]["max_den"]; total_hpwl = cost_info[-1]["total_hpwl"]
        #####################################################################################################################################
        # Check early stop
        enoughIter = iter >= cfg['ENOUGH_ITER']
        spreadEnough = total_density_overflow <= cfg['target_density'] + cfg['SpreadEnoughMoreDen']
        increaseOverDen = total_density_overflow >= total_over_den_prev - 1e-4
        increaseMaxDen = max_den >= max_den_prev - 1e-4
        notEfficientOptimize = 0.5 * total_potential_overflow * weight_density / total_obj_value * 100.0 > 95
        if enoughIter and notEfficientOptimize:
            not_efficient = True
            early_stop = True
        if enoughIter and spreadEnough and increaseOverDen and increaseMaxDen:
            early_stop = True
        #  update density weight (and recalculate cost)
        weight_density = placer.update_weight_density(iter)
        
        if early_stop:
            break
    if (bPlot):
        placer.save_gif("gp.gif")

    # Visualization Check 
    start = time.time()
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), facecolor='white')
    total_lse = placer.calculate_lse_wl()
    total_hpwl = placer.calculate_hpwl()
    total_density_overflow, _, max_den = placer.calculate_overflow_density()
    total_potential_overflow, _, _ = placer.calculate_overflow_potential()
    grad_wl = placer.calculate_wl_gradient(); grad_pot = placer.calculate_density_gradient()
    wl_pot_balance_ratio = total_lse / total_potential_overflow
    total_obj_value = placer.calculate_obj_value(total_lse, total_potential_overflow, weight_density * wl_pot_balance_ratio)
    costs_info = {"total_hpwl": total_hpwl, "total_lse": total_lse.item(), "total_density_overflow": total_density_overflow, "total_potential_overflow": total_potential_overflow.item(), "total_obj_value": total_obj_value.item(), "total_cost": total_obj_value.item(), "weight_density": weight_density.item(), "gWL": np.sum(np.abs(grad_wl.detach().cpu().numpy())), "gPot": np.sum(np.abs(grad_pot.detach().cpu().numpy())), "max_den": max_den, "wl_pot_balance_ratio": wl_pot_balance_ratio.item()}
    density_map = placer.bins_potential.detach().cpu().numpy() + placer.bins_base_potential.detach().cpu().numpy()
    overflow_map = np.maximum(placer.bins_potential.detach().cpu().numpy() - placer.bins_expect_potential.detach().cpu().numpy(), 0)
    placer.plot_placement(axs[0], 0, 0, costs_info, draw_net=False, show_name=False)
    placer.plot_density_map(fig, axs[1], "Density Map", density_map, _vmax=2)
    placer.plot_density_map(fig, axs[2], "Overflow Map", overflow_map, _vmax=2)
    plt.savefig("gp.png")
    end = time.time()
    print(f"Plot_placement, Time taken: {end - start:.3f} seconds")
    


    end = time.time()
    print(f"Analytical_placement, Time taken: {end - start:.3f} seconds")



if __name__ == "__main__":
  main()