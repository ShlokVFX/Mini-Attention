import torch
import torch.nn.functional as F


class OnlineSoftmax:
    def __init__(self, device='cpu', dtype=torch.float32):
        # Initialize running max and exp sum
        self.max_val = None
        self.sum_exp = None
        self.device = device
        self.dtype = dtype

    def update(self, x):
        """
        Update running softmax statistics in a numerically stable way.
        """
        x = x.to(device=self.device, dtype=self.dtype)

        # Compute max of current block
        block_max = torch.max(x, dim=-1, keepdim=True).values

        # First update case
        if self.max_val is None:
            self.max_val = block_max
            exp_x = torch.exp(x - self.max_val)
            self.sum_exp = exp_x.sum(dim=-1, keepdim=True)
            return

        # Compute new global max
        new_max = torch.maximum(self.max_val, block_max)

        # Rescale previous sum_exp
        self.sum_exp = self.sum_exp * torch.exp(self.max_val - new_max)

        # Add current block contribution
        exp_x = torch.exp(x - new_max)
        self.sum_exp = self.sum_exp + exp_x.sum(dim=-1, keepdim=True)

        # Update stored max
        self.max_val = new_max

    def compute(self, x):
        """
        Compute normalized softmax values using stored statistics.
        """
        x = x.to(device=self.device, dtype=self.dtype)
        exp_x = torch.exp(x - self.max_val)
        return exp_x / self.sum_exp


def test_online_softmax():
    x = torch.randn(2, 8)
    print("Input shape:", x.shape)

    # Standard softmax
    standard_softmax = F.softmax(x, dim=-1)

    # Online softmax (simulate streaming columns)
    online_softmax = OnlineSoftmax(device=x.device, dtype=x.dtype)

    for i in range(x.size(1)):
        online_softmax.update(x[:, i:i+1])

    online_result = online_softmax.compute(x)

    print("Standard softmax (first 4 elements):", standard_softmax[0, :4])
    print("Online softmax (first 4 elements):", online_result[0, :4])
    print(
        "Maximum difference:",
        torch.max(torch.abs(standard_softmax - online_result)).item()
    )


if __name__ == "__main__":
    test_online_softmax()