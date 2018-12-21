class DICELoss(Function):
    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, input_, target):
        self.save_for_backward(input_, target)
        eps = 0.000001
        _, result_ = input_.max(1)
        result_ = torch.squeeze(result_)

        result = torch.cuda.FloatTensor(result_.size())
        self.target_ = torch.cuda.FloatTensor(target.size())

        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
        target.unsqueeze_(-1)
        target = target.expand(result.size())
        intersect = result * target

        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        IoU = intersect / union
        out = torch.FloatTensor(1).fill_(2*IoU)
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input_, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect / (union*union)
        pred = torch.mul(input_[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(torch.unsqueeze(dDice,1), grad_output[0]),
                                torch.mul(torch.unsqueeze(dDice,1), -grad_output[0])), dim = 1)
        return grad_input, None