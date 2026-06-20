import torch
import torch.optim as optim
import numpy as np
from utils import rollout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50,
                       batch_size=32):

    # Hint: Use standard pytorch supervised learning code to train the policy.
    optimizer = optim.Adam(list(policy.parameters()), lr=1e-4)
    all_obs = np.concatenate([d['observations'] for d in expert_data]) # obs are the same as states
    all_actions = np.concatenate([d['actions'] for d in expert_data])
    num_samples = all_obs.shape[0]
    num_batches = num_samples // batch_size
    losses = []
    for epoch in range(num_epochs):
        ## TODO Students
        order = np.arange(num_samples)
        np.random.shuffle(order)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            #========== TODO: start ==========
            # Forward Pass
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            X_batch = torch.tensor(all_obs[start_idx:end_idx], dtype=torch.float32)
            X_batch = X_batch.to(device)

            y_batch = torch.tensor(all_actions[start_idx:end_idx], dtype=torch.float32)
            y_batch = y_batch.to(device)

            # Gives the log likelihood of experts action
            ac_log_prob = policy.log_prob(X_batch, y_batch)
            loss = -(ac_log_prob).mean()

            # Backward Pass
            #========== TODO: end ==========
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # if epoch % 10 == 0:
        print('[%d] loss: %.8f' %
            (epoch, running_loss / num_batches))
        losses.append(loss.item())

    # Added return loss for plotting
    return losses