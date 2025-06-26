
from tqdm import tqdm
import torch
device='cuda'
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=3, checkpoint_path=None):
    best_loss = float('inf')
    best_model_state = None
    counter = 0
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for visual, audio, text, labels in loop:
            visual = visual.to(device)
            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(visual, audio, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * visual.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

        val_loss, val_acc = evaluate_loss(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            counter = 0
            if checkpoint_path:
                torch.save(best_model_state, checkpoint_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for visual, audio, text, labels in dataloader:
            visual = visual.to(device)
            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)

            outputs = model(visual, audio, text)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * visual.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for visual, audio, text, labels in dataloader:
            visual = visual.to(device)
            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)

            outputs = model(visual, audio, text)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * visual.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for visual, audio, text, labels in tqdm(dataloader, desc="Evaluating"):
            visual = visual.to(device)
            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)

            outputs = model(visual, audio, text)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = 100. * correct / total
    print(f"✅ Final Accuracy on validation set: {accuracy:.2f}%")
    return accuracy, all_predicted, all_labels
