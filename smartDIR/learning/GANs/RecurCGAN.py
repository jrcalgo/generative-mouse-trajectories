import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Conditional Generator
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, condition_dim, sequence_length):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.lstm1 = nn.LSTM(latent_dim + condition_dim, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, z, condition):
        z = torch.cat((z, condition), dim=1)
        z = z.unsqueeze(1).repeat(1, self.sequence_length, 1)
        x, _ = self.lstm1(z)
        x, _ = self.lstm2(x)
        output = torch.tanh(self.fc(x))
        return output

# Conditional Discriminator
class ConditionalDiscriminator(nn.Module):
    def __init__(self, condition_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.lstm1 = nn.LSTM(2 + condition_dim, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, condition):
        condition = condition.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat((x, condition), dim=2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        output = self.sigmoid(x)
        return output

class RecurrentCGAN:
    def __init__(self, latent_dim, condition_dim, sequence_length, lr=0.0002, batch_size=64, epochs=10000):
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sequence_length = sequence_length
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.generator = ConditionalGenerator(latent_dim, condition_dim, sequence_length)
        self.discriminator = ConditionalDiscriminator(condition_dim)

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()

    def train(self):
        for epoch in range(self.epochs):
            # Train Discriminator
            real_sequences = torch.tensor(np.random.normal(size=(self.batch_size, self.sequence_length, 2)), dtype=torch.float32)
            real_labels = torch.ones((self.batch_size, 1))
            fake_labels = torch.zeros((self.batch_size, 1))
            condition = torch.tensor(np.random.normal(size=(self.batch_size, self.condition_dim)), dtype=torch.float32)

            noise = torch.tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim)), dtype=torch.float32)
            generated_sequences = self.generator(noise, condition)

            self.d_optimizer.zero_grad()
            real_outputs = self.discriminator(real_sequences, condition)
            d_loss_real = self.criterion(real_outputs, real_labels)
            fake_outputs = self.discriminator(generated_sequences.detach(), condition)
            d_loss_fake = self.criterion(fake_outputs, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            self.d_optimizer.step()

            # Train Generator
            self.g_optimizer.zero_grad()
            fake_outputs = self.discriminator(generated_sequences, condition)
            g_loss = self.criterion(fake_outputs, real_labels)
            g_loss.backward()
            self.g_optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch} / {self.epochs} - D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    def predict(self, start_pos, end_pos):
        with torch.no_grad():
            noise = torch.tensor(np.random.normal(0, 1, (1, self.latent_dim)), dtype=torch.float32)
            condition = torch.tensor(np.concatenate((start_pos, end_pos)), dtype=torch.float32).unsqueeze(0)
            generated_sequence = self.generator(noise, condition)

            # Adjust generated sequence to interpolate between start and end positions
            generated_sequence = generated_sequence.numpy()[0]
            generated_sequence[:, 0] = start_pos[0] + (end_pos[0] - start_pos[0]) * (generated_sequence[:, 0] + 1) / 2
            generated_sequence[:, 1] = start_pos[1] + (end_pos[1] - start_pos[1]) * (generated_sequence[:, 1] + 1) / 2
        return generated_sequence

# Hyperparameters
latent_dim = 100
condition_dim = 4  # Start and end positions (x1, y1, x2, y2)
sequence_length = 50
lr = 0.0002
batch_size = 64
epochs = 10000

# Instantiate and train RecurrentCGAN
cgan = RecurrentCGAN(latent_dim, condition_dim, sequence_length, lr, batch_size, epochs)
cgan.train()

# Predict with the trained model
start_pos = np.array([0.0, 0.0])
end_pos = np.array([1.0, 1.0])
generated_sequence = cgan.predict(start_pos, end_pos)
print("Generated Mouse Trajectory:", generated_sequence)
