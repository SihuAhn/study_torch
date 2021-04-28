from model import *

state_dict = torch.load('our_model.pth')
print(state_dict.keys())
model.load_state_dict(state_dict)

# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = fc_model.Network(checkpoint['input_size'],
#                              checkpoint['output_size'],
#                              checkpoint['hidden_size'])
#     model.load_state_dict(checkpoint['state_dict'])
#
# model = load_checkpoint('our_model.pth')
# print(model)