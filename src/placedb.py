import numpy as np
import math

# PRINT_INFO = False

def Message(msg):
    print(f"[Placedb] - {msg}")

class Netlist():
    def __init__(self):
        # Cells   
        self.cells_name       = []
        self.cells_size       = []
        self.cells_pos        = [] # pos is center of cell
        self.cells_isfixed    = [] # 1: fixed, 0: movable
        self.cells_rotate     = []
        self.cells_name_to_id = {}
        self.cells_nets_list  = []
        # Nets
        self.nets             = []
        self.nets_pins_offset = []
        self.nets_weight      = []
class ChipInfo():
    """Defines the netlist structure"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.board_size     = (0, 0)
        self.board_llxy     = (0, 0)
        self.num_bins       = cfg['num_bins']
        self.bin_size       = (0, 0)
        self.target_density = cfg['target_density']
        self.netlist        = None
        self.n_movable      = 0
        self.n_terminal     = 0
        self.n_cell         = 0
        self.n_macro        = 0
        self.n_net          = 0
    
    def set_from_parser(self, parser):
        self.case_name = parser.designName
        self.parser_fileDir = parser._fileDir
        self.parser_auxName = parser._auxName
        self.parser_plName = parser._plName
        self.parser_nodesName = parser._nodesName
        self.parser_netsName = parser._netsName
        self.parser_wtsName = parser._wtsName
        self.parser_sclName = parser._sclName

        # ChipInfo
        outline_w = max([int(parser.bsRowList[i][3][2]) * int(parser.bsRowList[i][7][5]) for i in range(len(parser.bsRowList))]) # max Sitewidth * NumSites
        outline_h = len(parser.bsRowList) * int(parser.bsRowList[0][2][2]) # NumRows * Height
        outline_x = min([int(parser.bsRowList[i][7][2]) for i in range(len(parser.bsRowList))]) # min SubrowOrigin
        outline_y = min([int(parser.bsRowList[i][1][2]) for i in range(len(parser.bsRowList))]) # min Coordinate
        self.bin_size = (outline_w/self.num_bins, outline_h/self.num_bins)
        self.board_size = (outline_w, outline_h)
        self.board_llxy = (outline_x, outline_y)
        self.n_movable = parser.numInsts
        self.n_terminal = parser.numFixedInsts
        self.n_cell = self.n_movable + self.n_terminal
        self.n_net = len(parser.bsNetList)
        self.netlist = Netlist()

        # Blocks
        self.netlist.cells_name = [""] * (self.n_cell)
        self.netlist.cells_size = np.zeros((self.n_cell, 2))
        self.netlist.cells_pos = np.zeros((self.n_cell, 2))
        self.netlist.cells_rotate = [""] * (self.n_cell)
        self.netlist.cells_isfixed = np.zeros(self.n_cell)
        self.cells_name_to_id = {} # name: id
        self.netlist.cells_nets_list = [[] for i in range(self.n_cell)]

        self.n_macro = 0
        for inst in parser.bsInstList:
            if len(inst) == 3 and (round(float(inst[2])) > int(parser.bsRowList[0][2][2])): # instance and size.y > row-height
                self.n_macro += 1
        ## set id for name and size (macro, std-cell, fixed terminal)
        count_inst, count_fixInst = 0, 0
        count_macro, count_stdcell = 0, 0
        for inst in parser.bsInstList:
            c_id = 0
            if len(inst) == 3: # instance
                if round(float(inst[2])) > int(parser.bsRowList[0][2][2]): # macro
                    c_id = count_macro
                    count_macro += 1
                else: # std-cell
                    c_id = self.n_macro+count_stdcell
                    count_stdcell += 1
            else: # fixed terminal
                c_id = self.n_movable+count_fixInst
                self.netlist.cells_isfixed[c_id] = 1
                count_fixInst += 1
            self.cells_name_to_id[inst[0]] = c_id
            self.netlist.cells_name[c_id] = inst[0]
            self.netlist.cells_size[c_id] = [float(inst[1]), float(inst[2])]
        for pl in parser.bsPlList:
            self.netlist.cells_pos[self.cells_name_to_id[pl[0]]] = np.array([float(pl[1])-outline_x, float(pl[2])-outline_y])
            self.netlist.cells_rotate[self.cells_name_to_id[pl[0]]] = pl[4]

        # Netlist
        self.netlist.nets = []
        self.netlist.nets_pins_offset = []
        self.netlist.nets_weight = []
        net_id = 0
        for net in parser.bsNetList:
            cell_id_list = [self.cells_name_to_id[pin[0]] for pin in net[1]]
            cell_id_list = list(set(cell_id_list))
            self.netlist.nets.append(np.array(cell_id_list))
            self.netlist.nets_weight.append(1)
            pinoffsets = []
            for pin in net[1]:
                if len(pin) < 5: pinoffset = [0, 0] # terminal
                else: pinoffset = [float(pin[3]), float(pin[4])] # component
                pinoffsets.append(pinoffset)
            self.netlist.nets_pins_offset.append(np.array(pinoffsets))
            for c_id in cell_id_list:
                if net_id not in self.netlist.cells_nets_list[c_id]:
                    self.netlist.cells_nets_list[c_id].append(net_id)
            net_id += 1

    def print_info(self):
        Message(f"case_name:      {self.case_name}")
        Message(f"board_size:     {self.board_size}")
        Message(f"board_llxy:     {self.board_llxy}")
        Message(f"num_bins:       {self.num_bins}")
        Message(f"bin_size:       {self.bin_size}")
        Message(f"target_density: {self.target_density}")
        Message(f"#module:        {self.n_cell}")
        Message(f"#movable:       {self.n_movable}")
        Message(f"#std-cell:      {self.n_cell-self.n_terminal-self.n_macro}")
        Message(f"#macro:         {self.n_macro}")
        Message(f"#pad:           {self.n_terminal}")
        Message(f"#net:           {self.n_net}")
        board_area = self.board_size[0] * self.board_size[1]
        fixed_area = sum([self.netlist.cells_size[i][0] * self.netlist.cells_size[i][1] for i in range(self.n_cell) if self.netlist.cells_isfixed[i] == 1])
        movable_area = sum([self.netlist.cells_size[i][0] * self.netlist.cells_size[i][1] for i in range(self.n_cell) if self.netlist.cells_isfixed[i] == 0])
        Message(f"Util (%):       {movable_area/(board_area-fixed_area)*100:.2f}")