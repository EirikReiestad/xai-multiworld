import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression


class ProbeObservation:
    def __init__(self, probe: LogisticRegression) -> None:
        self._probe = probe

    def _get_positive_activations(self, activation_size: torch.Size):
        coef = torch.tensor(self._probe.coef_, dtype=torch.float32, requires_grad=True)
        intercept = torch.tensor(
            self._probe.intercept_, dtype=torch.float32, requires_grad=True
        )

        def objective_function(x):
            x = x.view(-1)
            logits = torch.matmul(x, coef.T) + intercept
            prob_positive = torch.sigmoid(logits)
            return -prob_positive

        initial_point = torch.zeros(activation_size, requires_grad=True)

        optimizer = optim.LBFGS([initial_point], lr=0.1, max_iter=100)

        def closure():
            optimizer.zero_grad()
            loss = objective_function(initial_point)
            loss.backward()
            return loss

        optimizer.step(closure)

        optimal_point = initial_point.detach().numpy()
        optimal_point_probability = self._probe.predict_proba([optimal_point])[0, 1]

        print("Optimized point classified as positive:", optimal_point)
        print(
            "Probability of the positive class for the optimized point:",
            optimal_point_probability,
        )
