import torch.nn as nn
import torch


class CenterLoss(nn.Module):
    def __init__(self, num_class, vector_size):
        super(CenterLoss, self).__init__()
        self.num_class = num_class
        self.centers = nn.Parameter(torch.randn(self.num_class, vector_size))
        # self.centers = nn.Parameter(torch.ones((self.num_class, vector_size)))
        self.save_for_backward = {}

    def forward(self, target, vector_embedding):
        center_by_target = self.centers.index_select(0, target.long())

        diff_vector_from_center = vector_embedding - center_by_target
        self.save_for_backward = diff_vector_from_center
        
        # check with notebook at home about i,j,yi
        return 1/2 * torch.pow(diff_vector_from_center, 2).sum()

    def backward(self):
        diff_vector_from_center = self.save_for_backward
        # -diff_vector_from_center 
        counts = self.centers.new_zeros(self.centers.size())
        ones = self.centers.new_ones(self.centers.size())

        print('counts:', counts)
        print('ones:', ones)


def main():
    print('-'*80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.Tensor([0, 1, 2, 3]).to(device)
    vector_embedding = torch.tensor([[10, 10],
                                     [11, 11],
                                     [12, 12],
                                     [13, 13]], dtype=torch.float).to(device).requires_grad_()
    ct_loss = CenterLoss(10, vector_embedding.shape[1]).to(device)
    # print (list(ct.parameters()))
    optimizer = torch.optim.SGD(ct_loss.parameters(), lr=0.5)


    for i in range(100):
        out = ct_loss(target, vector_embedding)
        print('out:', out)
        out.backward()
        optimizer.step()
        optimizer.zero_grad()
        # with torch.no_grad():
        #     ct_loss.centers -= 1*ct_loss.centers.grad
        #     # print('a ct_loss.centers:', ct_loss.centers.grad)
        #     ct_loss.centers.grad.zero_()
    # print(ct.centers.grad)
    # print(feat.grad)


if __name__ == '__main__':
    torch.manual_seed(999)
    main()
