from pathlib import Path

from dr_fifo import FIFO_solver
from dr_mwtr import MWTR_solver
from dr_spt import SPT_solver
from dr_lpt import LPT_solver

base_path = Path(__file__).parent.absolute() / "instances/test"
instances = []
for i in range(1000):
    instances.append({"instance_path": base_path / ("t" + str(i))})

fifo = []
mwtr = []
spt = []
lpt = []

for instance in instances:
    fifo.append(FIFO_solver(instance))
    mwtr.append(MWTR_solver(instance))
    spt.append(SPT_solver(instance))
    lpt.append(LPT_solver(instance))

print(instances[0])
print("fifo", sum(fifo) / len(fifo))
print("mwtr", sum(mwtr) / len(mwtr))
print("spt", sum(spt) / len(spt))
print("lpt", sum(lpt) / len(lpt))