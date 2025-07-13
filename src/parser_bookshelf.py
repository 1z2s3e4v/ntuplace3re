import os
from os import path as os_path
import numpy as np
from enum import Enum
from collections import Counter

from util import bcolors

PRINT_INFO = False

def ErrorQuit(msg):
    print(f"[Parser] - {bcolors.FAIL}ERROR: {msg}{bcolors.ENDC}")
    exit(1)
def Message(msg):
    if PRINT_INFO:
        print(f"[Parser] - {msg}")

# bookshelf object type
class BsType (Enum):
    MOVABLE_STD_INST = 0
    MOVABLE_MACRO_INST = 1
    FIXED_INST = 2
    PRIMARY = 3

class Parser():
    def __init__(self, auxName=None):
        if auxName == None:
            return
        self._fileDir = os.path.dirname(auxName)
        self._auxName = auxName
        self._plName = ""
        self._sclName = ""
        self._netsName = ""
        self._nodesName = ""
        self._wtsName = ""
        self._shapesName = ""
        self._routeName = ""

        self.designName = os_path.basename(auxName).split(".")[0]
        # nodes
        self.bsInstList = []
        self.numNodes = 0
        self.numTerminals = 0
        self.numInsts = 0
        self.numFixedInsts = 0
        self.numPrimary = 0
        # nets
        self.bsNetList = []
        # pl
        self.bsPlList = []
        # scl
        self.bsRowList = []

        # clustering map
        self.bsPartList = []

    def parse_bookshelf(self, print_info=False):
        global PRINT_INFO; PRINT_INFO = print_info
        self.parse_aux()
        self.parse_nodes()
        self.parse_pl(self._plName)
        self.parse_nets()
        self.parse_scl()
        # The *.shape is optional!
        if self._shapesName != "":
            self.parse_shapes()
        if self._routeName != "":
            self.parse_routes()
        Message(f"{bcolors.OKGREEN}Bookshelf Parsing Done!{bcolors.ENDC}")

    def parse_aux(self, print_info=False):
        Message(f"Parsing {self._auxName} ..." )
        f = open(self._auxName, 'r')
        cont = f.read()
        f.close()

        for curLine in cont.split("\n"):
            # should skip 1st, 2nd elements
            for curFile in curLine.strip().split()[2:]:
                if curFile.endswith("nodes"):
                    self._nodesName = os_path.join(self._fileDir, curFile)
                elif curFile.endswith("nets"):
                    self._netsName = os_path.join(self._fileDir, curFile)
                elif curFile.endswith("shapes"):
                    self._shapesName = os_path.join(self._fileDir, curFile)
                elif curFile.endswith("scl"):
                    self._sclName = os_path.join(self._fileDir, curFile)
                elif curFile.endswith("pl"):
                    self._plName = os_path.join(self._fileDir, curFile)
                elif curFile.endswith("route"):
                    self._routeName = os_path.join(self._fileDir, curFile)
                elif curFile.endswith("wts"):
                    self._wtsName = os_path.join(self._fileDir, curFile)
                    #print("[WARNING] *.wts will be ignored")
        if self._nodesName == "":
            ErrorQuit("*.nodes is missing")
        if self._netsName == "":
            ErrorQuit("*.nets is missing")
        if self._sclName == "":
            ErrorQuit("*.scl is missing")
        if self._plName == "":
            ErrorQuit("*.pl is missing")
    def parse_nodes(self):
        Message(f"Parsing {self._nodesName} ...")
        f = open(self._nodesName, 'r')
        cont = f.read()
        f.close()

        InstList = []
        for curLine in cont.split("\n"):
            curLine = curLine.strip()
            if curLine.startswith("UCLA") or curLine.startswith("#") or curLine == "":
                continue

            if curLine.startswith("NumNodes"):
                self.numNodes = int(curLine.split(" ")[-1])
            elif curLine.startswith("NumTerminals"):
                self.numTerminals = int(curLine.split(" ")[-1])
            else:
                InstList.append(curLine.split())

        Message(f"  From Nodes: NumTotalNodes:  {self.numNodes} ...")
        Message(f"  From Nodes: NumTerminals:  {self.numTerminals} ...")

        self.numInsts = 0
        self.numFixedInsts = 0
        self.numPrimary = 0
        self.bsInstList = []
        movableList = []
        terminalList = []
        for curInst in InstList:
            if len(curInst) == 3:
                self.numInsts += 1
                movableList.append(curInst)
            elif curInst[-1] == "terminal":
                self.numFixedInsts += 1
                terminalList.append(curInst)
            elif curInst[-1] == "terminal_NI":
                self.numPrimary += 1
                terminalList.append(curInst)
        self.bsInstList = movableList + terminalList

        Message(f"  Parsed Nodes: NumInsts: {self.numInsts}")
        Message(f"  Parsed Nodes: NumFixedInsts: {self.numFixedInsts}")
        Message(f"  Parsed Nodes: NumPrimary: {self.numPrimary}")

    def parse_nets(self):
        Message(f"Parsing {self._netsName} ...")
        f = open(self._netsName, 'r')
        cont = f.read()
        f.close()

        tmpNetlist = []
        for curLine in cont.split("\n"):
            curLine = curLine.strip()
            if curLine.startswith("UCLA") or curLine.startswith("#") or curLine == "":
                continue

            if curLine.startswith("NumNets"):
                numNets = int(curLine.split(" ")[-1])
            elif curLine.startswith("NumPins"):
                numPins = int(curLine.split(" ")[-1])
            else:
                tmpNetlist.append(curLine.split())

        Message(f"  From Nets: NumNets:  {numNets} ...")
        Message(f"  From Nets: NumPins:  {numPins} ...")

        self.bsNetList = []
        count_net = 0
        for idx, curArr in enumerate(tmpNetlist):
            if curArr[0] == "NetDegree":
                numPinsInNet = int(curArr[2])
                netName = "net"+str(count_net)
                if len(curArr) > 3:
                    netName = curArr[3]
                pinArr = [l for l in tmpNetlist[(idx+1):(idx+numPinsInNet+1)]]
                self.bsNetList.append([netName, pinArr])
                count_net += 1

        Message(f"  Parsed Nets: NumNets: {len(self.bsNetList)}")

    def parse_pl(self, file_name):
        Message(f"Parsing {file_name} ...")
        f = open(file_name, 'r')
        cont = f.read()
        f.close()

        self.bsPlList = []
        for curLine in cont.split("\n"):
            curLine = curLine.strip()
            if curLine.startswith("UCLA") or curLine.startswith("#") or curLine == "":
                continue
            self.bsPlList.append(curLine.split())
        Message(f"  Parsed pl: NumInstsInPl: {len(self.bsPlList)}")

        numFixedInsts = [0 for curArr in self.bsPlList if curArr[-1] == "/FIXED"]
        numPrimary = [0 for curArr in self.bsPlList if curArr[-1] == "/FIXED_NI"]

        Message(f"  Parsed pl: NumFixedInsts: {len(numFixedInsts)}")
        Message(f"  Parsed pl: NumPrimary: {len(numPrimary)}")

    def parse_scl(self):
        Message(f"Parsing {self._sclName} ...")
        f = open(self._sclName, 'r')
        cont = f.read()
        f.close()

        tmpRowList = []
        for curLine in cont.split("\n"):
            curLine = curLine.strip()
            if curLine.startswith("UCLA") or curLine.startswith("#") or curLine == "":
                continue

            if curLine.startswith("NumRows") or curLine.startswith("Numrows"):
                numRows = int(curLine.split(" ")[-1])
            else:
                tmpRowList.append(curLine.split())
        Message(f"  From scl: NumRows: {numRows}")

        # extract indices on CoreRow/End
        coreRowIdxList = [idx for idx,tmpArr in enumerate(tmpRowList) if tmpArr[0] == "CoreRow"]
        endIdxList = [idx for idx,tmpArr in enumerate(tmpRowList) if tmpArr[0] == "End"]

        if len(coreRowIdxList) != len(endIdxList):
            ErrorQuit("The number of CoreRow and End is different in scl!")

        self.bsRowList = []
        for idx1, idx2 in zip(coreRowIdxList, endIdxList):
            self.bsRowList.append(tmpRowList[idx1:idx2])
        Message(f"  Parsed scl: NumRows: {len(self.bsRowList)}")
    
    def parse_shapes(self):
        pass
    def parse_routes(self):
        pass

    def print_parser_info(self):
        Message(f"Parser Information:")
        Message(f"  Design Name: {self.designName}")
        Message(f"  NumNodes: {self.numNodes}")
        Message(f"  NumTerminals: {self.numTerminals}")
        Message(f"  NumInsts: {self.numInsts}")
        Message(f"  NumFixedInsts: {self.numFixedInsts}")
        Message(f"  NumPrimary: {self.numPrimary}")
        Message(f"  NumNets: {len(self.bsNetList)}")
        Message(f"  NumPl: {len(self.bsPlList)}")

def parse_clustering_map(file_name):
    Message(f"Parsing {file_name} ...")
    bsPartList = []
    cell_nums = [0]
    if (not os.path.exists(file_name)):
        Message(f"Parsing {file_name} ... no file")
        return bsPartList, cell_nums
    f = open(file_name, 'r')
    cont = f.read().splitlines()
    f.close()
    bsPartList = list(map(int, cont))

    cluster_num = max(bsPartList) + 1
    cell_nums = Counter(bsPartList)
    cell_nums = [cell_nums.get(i, 0) for i in range(cluster_num)]

    # Message(f"  Parsed .part: NumInstsInPart: {len(bsPartList)}")
    # Message(f"  Parsed .part: NumCluster: {cluster_num}")
    # Message(f"{bcolors.OKGREEN}Clustering-Map Parsing Done!{bcolors.ENDC}")
    return bsPartList, cell_nums

