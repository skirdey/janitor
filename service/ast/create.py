from os import path
from wget import download
from torch import load, jit
from model import ASTModel

model = ASTModel()
model.eval()

checkpoint_path = "./pretrained_models/audioset_10_10_0.4593.pth"
if not path.exists("./pretrained_models/audioset_10_10_0.4593.pth"):
    audioset_mdl_url = (
        "https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1"
    )
    download(
        audioset_mdl_url,
        out=checkpoint_path,
    )
checkpoint = load(checkpoint_path, weights_only=True)
state_dict = {
    key.replace("module.", ""): value for key, value in checkpoint.items()
}
model.load_state_dict(state_dict)

torch_script_module = jit.script(model)
torch_script_module.save("./model.pt")
