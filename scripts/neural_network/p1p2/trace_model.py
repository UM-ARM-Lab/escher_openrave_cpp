import torch, IPython

from model_B_0 import Model

def main():
    model = Model()
    filename = 'depth_and_boundary_combined_B_0_weighted_0.001_checkpoint/epoch=20.checkpoint.pth.tar'
    checkpoint = torch.load(filename, map_location=lambda storage, loc:storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to('cpu')
    model.eval()
    print('Successfully load the model')
    traced_script_module = torch.jit.trace(model, (torch.rand(1,1,65,65), torch.rand(1,1,25,252), torch.rand(1,3)))
    print(model.forward(torch.ones(1,1,65,65) * 2, torch.ones(1,1,25,252) * 2, torch.ones(1,3) * 2))
    # traced_script_module.save('depth_and_boundary_combined_B_0_weighted_0.001.pt')




if __name__ == '__main__':
    main()
