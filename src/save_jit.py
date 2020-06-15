import torch

from models import DBTextModel

dbnet = DBTextModel()
example_input = torch.rand(1, 3, 640, 640)
dbnet.eval()

traced_model = torch.jit.trace(dbnet, example_input)
traced_model.save("./models/db_resnet18_jit.pt")
