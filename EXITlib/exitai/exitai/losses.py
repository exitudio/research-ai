import torch.nn as nn
from torch.autograd.function import Function
import torch

'''
 - To compare the different Center Loss & autograd, set inital Param(center) to one rather than random
 - The things that is different is the paper add "1" for each class of center grad
'''
class CenterLossAutograd(nn.Module):
    def __init__(self, num_class, vector_size):
        super(CenterLossAutograd, self).__init__()
#         self.centers = nn.Parameter(torch.ones((num_class, vector_size)))
        self.centers = nn.Parameter(torch.randn(num_class, vector_size))
    def forward(self, target, vector_embedding):
        center_by_target = self.centers.index_select(0, target.long())
        diff_vector_from_center = vector_embedding - center_by_target
        batch_size = diff_vector_from_center.size(0)
        return 1/2 * diff_vector_from_center.pow(2).sum() / batch_size
    
class CenterLoss(nn.Module):
    def __init__(self, num_class, vector_size):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_class, vector_size))
#         self.centers = nn.Parameter(torch.ones((num_class, vector_size)))
        self.center_loss_function = CenterLossFunction.apply

    def forward(self, target, vector_embedding):
        return self.center_loss_function(target, vector_embedding, self.centers)


class CenterLossFunction(Function):
    @staticmethod
    def forward(ctx, target, vector_embedding, centers):
        center_by_target = centers.index_select(0, target.long())

        diff_vector_from_center = vector_embedding - center_by_target
        ctx.save_for_backward(diff_vector_from_center, target, centers)

        batch_size = diff_vector_from_center.size(0)
        return 1/2 * diff_vector_from_center.pow(2).sum() / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        diff_vector_from_center, target, centers = ctx.saved_tensors
        batch_size = diff_vector_from_center.size(0)

        # maybe try to fill 0.00000001
        counts = centers.new_ones((centers.size(0), 1)) # centers.new_full((centers.size(0), 1), 1e-8)
        ones = centers.new_ones((target.size(0), 1))

        counts = counts.scatter_add_(0, target.unsqueeze(1).long(), ones)
        grad_centers = centers.new_zeros(centers.size())
        grad_centers = grad_centers.scatter_add_(0, target.unsqueeze(1).expand(
            diff_vector_from_center.size()).long(), diff_vector_from_center)
        grad_centers = grad_centers/counts

        return None, grad_output * diff_vector_from_center / batch_size, -grad_centers / batch_size,


def main():
    print('-'*80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.Tensor([0, 0, 2, 3]).to(device)
    vector_embedding = torch.tensor([[10, 10],
                                     [11, 11],
                                     [12, 12],
                                     [13, 13]], dtype=torch.float).to(device).requires_grad_()
    ct_loss = CenterLoss(10, vector_embedding.shape[1]).to(device)
    ct_loss_autograd = CenterLossAutograd(10, vector_embedding.shape[1]).to(device)
    # print (list(ct.parameters()))
    optimizer = torch.optim.SGD(ct_loss.parameters(), lr=0.5)
    optimizer_test = torch.optim.SGD(ct_loss_autograd.parameters(), lr=0.5)

    for i in range(2):
        out = ct_loss(target, vector_embedding)
        out_test = ct_loss_autograd(target, vector_embedding)
        out.backward()
        out_test.backward()
        print('ct_loss:', ct_loss.centers.grad)
        print('ct_loss_autograd:', ct_loss_autograd.centers.grad)
        optimizer.step()
        optimizer.zero_grad()
        optimizer_test.step()
        optimizer_test.zero_grad()


if __name__ == '__main__':
    torch.manual_seed(999)
    main()
