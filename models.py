import torch
import torch.nn as nn
import torch.nn.functional as F
freq_res = 64


class FeatureExtractor(nn.Module):
    def __init__(self, freq_bins, n_classes, seq_len):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(freq_bins*seq_len, n_classes)
        
    def forward(self, x):
        # x shape: [batch, 1, seq_len, freq_bins]
        x = self.pool(F.relu(self.conv1(x)))  # shape: [batch, 16, seq_len/2, freq_bins/2]
        x = self.pool(F.relu(self.conv2(x)))  # shape: [batch, 32, seq_len/4, freq_bins/4]
        x = self.pool(F.relu(self.conv3(x)))  # shape: [batch, 64, seq_len/8, freq_bins/8]
        x = self.flatten(x)  # Flatten the tensor
        class_means = self.fc(x)  # shape: [batch, n_classes]
        class_var = self.fc(x)
        return class_means, class_var
    

class SpectrogramGenerator(nn.Module):
    def __init__(self, n_classes, seq_len, latent_dim=100, loudness_dim=1, hidden_dim=512, n_layers=3):
        super(SpectrogramGenerator, self).__init__()
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.gamma_layer = nn.Linear(n_classes, latent_dim)
        self.beta_layer = nn.Linear(n_classes, latent_dim)
        self.FeatureExtractor = FeatureExtractor(freq_bins=64, n_classes=n_classes, seq_len=seq_len)
        # self.class_means = nn.Parameter(torch.randn(n_classes))
        # self.class_mean_embedding = nn.Embedding(n_classes, 1)
        # self.class_mean_embedding.weight.data.fill_(1.0)
        
        # If using a single latent vector, expand it to match the sequence length
        self.fc_init = nn.Linear(latent_dim + loudness_dim, hidden_dim)

        # Define the RNN layer(s)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True)

        # Define the output layer
        self.fc_out = nn.Linear(hidden_dim, freq_res)

    def forward(self, z, loudness, Melspec=None, con_vec=None):
        seq_len = loudness.size(1)
    
        if Melspec is not None:
            mean_extracted, var_extracted = self.FeatureExtractor(Melspec)
            con_vec = torch.normal(mean=mean_extracted, std=var_extracted)
            #con_vec = torch.clip(con_vec, min=0, max=1)
        elif con_vec is None:
            raise ValueError("Either Melspec or con_vec must be provided")
        else:
            mean_extracted = None
            var_extracted = None

        gamma = self.gamma_layer(con_vec)  # shape: (batch, latent_dim)
        beta = self.beta_layer(con_vec)    # shape: (batch, latent_dim)
        modulated_latent = gamma * z + beta
        z = modulated_latent.unsqueeze(1).repeat(1, seq_len, 1)

        rnn_input = torch.cat((z, loudness), dim=2)
        # Process the initial input to match the RNN input features
        rnn_input = self.fc_init(rnn_input)

        # Get the RNN output for the whole sequence
        rnn_out, _ = self.rnn(rnn_input)

        # Process the RNN outputs to the final spectrogram shape
        output_seq = self.fc_out(rnn_out.contiguous().view(-1, self.hidden_dim))
        spectrogram = output_seq.view(-1, 1, seq_len, freq_res)

        return spectrogram, mean_extracted, var_extracted


class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, n_classes*10),
            nn.Linear(n_classes*10, 25600)
        )
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(25600, 1)
        )

    def forward(self, img, labels):
        # Expand label embedding to same size as image
        seq_len = img.size(2)
        labels = self.label_embedding(labels)
        labels = labels.view(labels.size(0), 1, seq_len, freq_res)
        # Concatenate label and image
        d_in = torch.cat((img, labels), 1) # Concatenate along channel dimension
        print(d_in.shape)
        output = self.model(d_in)
        print(output.shape)
        return output
