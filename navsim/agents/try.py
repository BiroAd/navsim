from navsim.agents.transfuser.transfuser_model import TransfuserModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.agents.transfuser.transfuser_config import TransfuserConfig

from navsim.agents.vision_llm.vision_llm_model import VisionLLMModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.agents.vision_llm.vision_llm_config import VisionLLMConfig


config = VisionLLMConfig()
sampler = TrajectorySampling(time_horizon=4, interval_length=0.5)
model = VisionLLMModel(sampler, config)

print(model)