import torch
import torch.nn as nn
import math
class INR_Layer(nn.Module):
    def __init__(self, input_features, output_features, latent_dim = 64, is_first = False, bias = True, omega_0 = 30):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.latent_dim = latent_dim
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(self.input_features,self.output_features, bias=bias)

        self.scale = nn.Linear(in_features=self.latent_dim, out_features=self.output_features)
        self.shift = nn.Linear(in_features=self.latent_dim, out_features=self.output_features)

        self.init_weights()

    def init_weights(self):
        # initialization as described in the SIREN paper
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.input_features, 1 / self.input_features)
            else:
                bound = math.sqrt(6 / self.input_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x, latent):
        shift_modulation = self.shift(latent)
        shift_modulation = shift_modulation.unsqueeze(1)
        linear_output = self.omega_0 * self.linear(x) 
        modulated_linear_output = linear_output + shift_modulation
        
        if self.is_first:
            sine_output = torch.sin(linear_output)
            return sine_output
        else:
            sine_output = torch.sin(modulated_linear_output)
            return sine_output

class INR(nn.Module):
    def __init__(self,input_features=2, hidden_features = 256, output_features = 1, latent_dim = 64, omega_0 = 30):
        super().__init__()
        self.latent_dimensions = latent_dim
        
        self.INR_1 = INR_Layer(input_features=input_features, output_features=hidden_features,latent_dim=latent_dim, is_first=True, omega_0=omega_0)
        self.INR_2 = INR_Layer(input_features=hidden_features, output_features=hidden_features,latent_dim=latent_dim, is_first= False, omega_0=omega_0)
        self.INR_3 = INR_Layer(input_features=hidden_features, output_features=hidden_features,latent_dim=latent_dim, is_first= False, omega_0=omega_0)
        self.INR_4 = INR_Layer(input_features=hidden_features, output_features=hidden_features,latent_dim=latent_dim, is_first= False, omega_0=omega_0)
        self.output_layer = nn.Linear(hidden_features, output_features)

    def forward(self, x, image_modulation_vector):
        output_1 = self.INR_1(x, image_modulation_vector)
        output_2 = self.INR_2(output_1, image_modulation_vector)
        output_3 = self.INR_3(output_2, image_modulation_vector)
        output_4 = self.INR_4(output_3, image_modulation_vector)
        output = self.output_layer(output_4)
        return output

