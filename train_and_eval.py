
from tqdm import tqdm
import torch
device='cuda'

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for visual, audio, text, labels in loop:
            visual = visual.to(device)  # Shape: (batch_size, num_frames, C, H, W)
            audio = audio.to(device)    # Shape: (batch_size, n_mfcc, max_length)
            text = text.to(device)      # Shape: (batch_size, text_embedding_size)
            labels = labels.to(device)  # Shape: (batch_size)

            optimizer.zero_grad()

            outputs = model(visual, audio, text)  # Shape: (batch_size, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * visual.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), accuracy=100.*correct/total)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")


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
    print(f"Accuracy on the test set: {accuracy:.2f}%")
    return accuracy, all_predicted, all_labels
