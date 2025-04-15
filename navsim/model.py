from navsim.agents.transfuser.transfuser_model import TransfuserModel
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torchviz import make_dot
import torch
import torch.fx as fx
import graphviz

model = TransfuserModel(
    trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
    config=TransfuserConfig(),
)

print(model)

'''import hiddenlayer as hl

dummy_input = {
    "camera_feature": torch.randn(1, 3, 1024, 256),  # Example input shape
    "lidar_feature": torch.randn(1, 1, 256, 256),
    "status_feature": torch.randn(1, 8),
}

# Create a graph of the model
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/model_visualization")
writer.add_graph(model, (dummy_input,))
writer.close()'''

'''traced_model = fx.symbolic_trace(model)

g = graphviz.Digraph('Transfuser Model', filename='transfuser_fx_graph')
for node in traced_model.graph.nodes:
    g.node(str(node), f'{node.op}: {node.target}')
    for input_node in node.all_input_nodes:
        g.edge(str(input_node), str(node))
g.render()'''

# Print the model as a graph
#print(traced_model.graph)

# Gradients


'''dummy_input = {
    "camera_feature": torch.randn(1, 3, 1024, 256),  # Example input shape
    "lidar_feature": torch.randn(1, 1, 256, 256),
    "status_feature": torch.randn(1, 8),
}

# Forward pass to generate the computational graph
output = model(dummy_input)
bev_semantic_map = output["bev_semantic_map"]

# Visualize the model as a graph
dot = make_dot(bev_semantic_map, params=dict(model.named_parameters()))
dot.format = "png"
dot.render("transfuser_model")

print("Model visualization saved as 'transfuser_model.png'")'''