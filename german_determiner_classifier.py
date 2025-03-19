import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Define hyperparameters
lr = 0.1
max_epochs = 100
batch_size = 8  # Using mini-batch training

# Define input dimensions
num_classes = 6
num_features = 2 + 3 + 4  # Number: 2D, Gender: 3D, Case: 4D

# Define dataset mappings
number_map = {"sg": [1, 0], "pl": [0, 1]}
gender_map = {"masc": [1, 0, 0], "fem": [0, 1, 0], "neut": [0, 0, 1]}
case_map = {"nom": [1, 0, 0, 0], "acc": [0, 1, 0, 0], "dat": [0, 0, 1, 0], "gen": [0, 0, 0, 1]}

determiner_labels = {"der": 0, "die": 1, "das": 2, "den": 3, "dem": 4, "des": 5}

data = [("sg", "masc", "nom", "der"), ("sg", "masc", "acc", "den"),
        ("sg", "masc", "dat", "dem"), ("sg", "masc", "gen", "des"),
        ("sg", "fem", "nom", "die"), ("sg", "fem", "acc", "die"),
        ("sg", "fem", "dat", "der"), ("sg", "fem", "gen", "der"),
        ("sg", "neut", "nom", "das"), ("sg", "neut", "acc", "das"),
        ("sg", "neut", "dat", "dem"), ("sg", "neut", "gen", "des"),
        ("pl", "masc", "nom", "die"), ("pl", "fem", "nom", "die"),
        ("pl", "neut", "nom", "die"), ("pl", "masc", "acc", "die"),
        ("pl", "fem", "acc", "die"), ("pl", "neut", "acc", "die"),
        ("pl", "masc", "dat", "den"), ("pl", "fem", "dat", "den"),
        ("pl", "neut", "dat", "den"), ("pl", "masc", "gen", "der"),
        ("pl", "fem", "gen", "der"), ("pl", "neut", "gen", "der")]

# Prepare training data
inputs = [number_map[num] + gender_map[gen] + case_map[case] for num, gen, case, _ in data]
targets = [determiner_labels[det] for _, _, _, det in data]

torch_inputs = torch.tensor(inputs, dtype=torch.float32)
torch_targets = torch.tensor(targets, dtype=torch.long)

# Create DataLoader for batching
dataset = TensorDataset(torch_inputs, torch_targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the model
class DeterminerModel(nn.Module):
    def __init__(self):
        super(DeterminerModel, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)  # Single-layer network

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss function, and optimizer
model = DeterminerModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(max_epochs):
    total_loss = 0
    all_correct = True
    
    for i, (inp, target) in enumerate(zip(torch_inputs, torch_targets)):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(inp.unsqueeze(0))
        loss = criterion(output, target.unsqueeze(0))
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute prediction
        pred_label = torch.argmax(output, dim=1).item()
        correct_label = target.item()
        print(f"{data[i][0]} {data[i][1]} {data[i][2]} {correct_label} {pred_label}")
        
        if pred_label != correct_label:
            all_correct = False
    
    print(f"Loss: {total_loss:.4f}")
    
    if all_correct:
        print(f"All correct after {epoch + 1} iterations.")
        break

# Part B: Explanation of Namreg language experiment
# The loss converges slowly, and the model struggles to converge within 100 iterations.


# Part C: Optional bonus question
#The issue arises because “die” must now satisfy an additional inequality that was not present before, 
# conflicting with existing correct classifications. 
# This forces the model to shift weights in a way that disrupts prior learned relationships, 
# making convergence slower or impossible in a simple linear model.
# The model uses a linear transformation, meaning each input x is mapped to an output z_i:
#
#     z_i = w_i * x
#
# The correct determiner must have the highest score:
#
#     z_correct > z_j  for all j ≠ correct
#
# Initially, for German:
#
#     z_des > z_i  for all i ≠ des
#
# After modifying Namreg:
#
#     z_die > z_i  for all i ≠ die
#
# But "die" is already the correct determiner for:
#
#     z_die > z_der for fem, nom, sg
#     z_die > z_den for fem, acc, sg
#
# Since the model is linear, assigning "die" as the correct determiner for both cases results in:
#
#     w_die * x_masc_gen_sg > w_des * x_masc_gen_sg
#     w_die * x_fem_nom_sg > w_der * x_fem_nom_sg
#
# Because x_masc_gen_sg and x_fem_nom_sg are different, forcing "die" to be correct for both causes contradictions:
#
#     w_die * (x_masc_gen_sg - x_fem_nom_sg) > w_j * (x_masc_gen_sg - x_fem_nom_sg)
#
# This contradicts the linear separability condition, making learning slower or impossible.

