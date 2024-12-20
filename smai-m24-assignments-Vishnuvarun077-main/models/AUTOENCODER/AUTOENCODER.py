import numpy as np
from models.MLP import MLP
class AutoEncoder:
    def __init__(self, input_size, hidden_sizes, latent_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        
        # Encoder: MLP to reduce dimensions
        self.encoder = MLP(input_size=input_size, hidden_sizes=hidden_sizes + [latent_size], 
                           output_size=latent_size, activation='relu', learning_rate=learning_rate)
        
        # Decoder: MLP to reconstruct the input
        self.decoder = MLP(input_size=latent_size, hidden_sizes=hidden_sizes[::-1] + [input_size], 
                           output_size=input_size, activation='sigmoid', learning_rate=learning_rate)

    def fit(self, X, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                
                # Forward pass through encoder
                latent_representation = self.encoder.forward_propagation(X_batch)
                
                # Forward pass through decoder
                reconstructed = self.decoder.forward_propagation(latent_representation)
                
                # Compute loss (Mean Squared Error)
                loss = np.mean((X_batch - reconstructed) ** 2)
                
                # Backward pass through decoder
                decoder_gradients = self.decoder.backward_propagation(latent_representation, X_batch)
                
                # Backward pass through encoder
                encoder_gradients = self.encoder.backward_propagation(X_batch, latent_representation)
                
                # Update parameters for both encoder and decoder
                self.decoder.update_parameters(decoder_gradients)
                self.encoder.update_parameters(encoder_gradients)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def get_latent(self, X):
        return self.encoder.forward_propagation(X)

    def reconstruct(self, X):
        latent_representation = self.get_latent(X)
        return self.decoder.forward_propagation(latent_representation)