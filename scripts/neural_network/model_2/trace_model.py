import torch, IPython

from model_v0 import Model

def main():
    model = Model()
    filename = 'v0_weighted_0.001_checkpoint/epoch=3.checkpoint.pth.tar'
    checkpoint = torch.load(filename, map_location=lambda storage, loc:storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model = model.to('cpu')
    model.eval()
    print('Successfully load the model')
    traced_script_module = torch.jit.trace(model, (torch.rand(1,2,65,65), torch.rand(1,2,25,84), torch.rand(1,2,25,84), torch.rand(1,3)))
    print(model.forward(torch.ones(1,2,65,65) * 2, torch.ones(1,2,25,84) * 2, torch.ones(1,2,25,84) * 2, torch.ones(1,3) * 2))
    traced_script_module.save('v0_weighted_0.001.pt')




if __name__ == '__main__':
    main()
