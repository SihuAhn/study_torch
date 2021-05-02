from model import *
from data_loader import *

my_model = Classifier()
my_model.load_state_dict(torch.load('checkpoint.pth'))
print(my_model)

images, labels = next(iter(test_loader))
ps = torch.exp(my_model(images))
print(ps.shape)

top_p, top_class = ps.topk(1, dim=1)
print(top_class[:10, :])

equals = top_class == labels.view(*top_class.shape)

accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')