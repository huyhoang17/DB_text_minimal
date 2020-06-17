import torch

from models import DBTextModel

model_path = '/home/phan.huy.hoang/phh_workspace/DB_text_minimal/models/db_resnet18.pth'
device = 'cpu'
dbnet = DBTextModel().to(device)
dbnet.load_state_dict(torch.load(model_path, map_location=device))
assert not next(dbnet.parameters()).is_cuda
example_input = torch.rand(1, 3, 640, 640)
dbnet.eval()

traced_model = torch.jit.trace(dbnet, example_input)
traced_model.save("./models/db_resnet18_jit.pt")
