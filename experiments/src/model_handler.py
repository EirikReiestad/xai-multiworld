import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR


def train_or_load_model(
    model,
    model_name,
    train_loader,
    test_loader,
    lr,
    gamma,
    epochs,
    log_interval,
    dry_run,
    load_model,
):
    if not load_model:
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        for epoch in range(1, epochs + 1):
            train_model(
                log_interval, dry_run, model, None, train_loader, optimizer, epoch
            )
            test_model(model, None, test_loader)
            scheduler.step()
            torch.save(model.state_dict(), model_name)
    else:
        model.load_state_dict(torch.load(model_name))
        model.eval()
        test_model(model, None, test_loader)
    return model


def train_model(
    log_interval, dry_run, model, device, train_loader, optimizer, epoch, verbose=False
):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=False)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % log_interval == 0:
            if verbose:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
            if dry_run:
                break

    accuracy = correct / total
    return accuracy


def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * accuracy,
        )
    )
    return accuracy
