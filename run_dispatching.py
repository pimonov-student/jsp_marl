from pathlib import Path

from final_torchrl.dr_fifo import FIFO_solver
from final_torchrl.dr_mwtr import MWTR_solver
from final_torchrl.dr_spt import SPT_solver
from final_torchrl.dr_lpt import LPT_solver

base_path = Path(__file__).parent.absolute() / "instances"
instances = [
    {"instance_path": base_path / "ta40"},
    {"instance_path": base_path / "ta41"},
    {"instance_path": base_path / "ta42"},
    {"instance_path": base_path / "ta43"},
    {"instance_path": base_path / "ta44"},
    {"instance_path": base_path / "ta45"},
    {"instance_path": base_path / "ta46"},
    {"instance_path": base_path / "ta47"},
    {"instance_path": base_path / "ta48"},
    {"instance_path": base_path / "ta49"},
    {"instance_path": base_path / "ta50"},
]

for instance in instances:
    fifo_solution = FIFO_solver(instance)
    mwtr_solution = MWTR_solver(instance)
    spt_solution = SPT_solver(instance)
    lpt_solution = LPT_solver(instance)
    print("--- ", instance, "---")
    print("fifo: ", fifo_solution)
    print("mwtr: ", mwtr_solution)
    print("spt:  ", spt_solution)
    print("lpt:  ", lpt_solution)