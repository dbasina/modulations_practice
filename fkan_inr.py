import torch
import torch.nn as nn
import numpy as np

class FourierKANLayer_TorchLike(nn.Module):
    def __init__(self, inputDimensions, outputDimensions, numFourierFrequencyBands, addBias = True, smoothInitialization=False):
        super().__init__()
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions
        self.numFourierFrequencyBands = numFourierFrequencyBands
        self.smoothInitialization = smoothInitialization
        self.addBias = addBias

        # if input is (x,y) we want to produce (cos(x), cos(y), sin(x), sin(y)) and combine linearly for all frequencies
        # so since its all just addition, we can use nn.Linear instead of multi dimensional matrix operations.
        self.linear = nn.Linear(2*self.inputDimensions*self.numFourierFrequencyBands, self.outputDimensions, bias=self.addBias)
    
    def forward(self, input):

        inputShape = input.shape
        outputShape = inputShape[:-1] + (self.outputDimensions,)

        input = input.reshape(-1, self.inputDimensions)
        frequencies = torch.arange(1, self.numFourierFrequencyBands + 1, device=input.device).float()
        inputScaled = input.unsqueeze(-1) * frequencies
        cosineComponent = torch.cos(inputScaled)
        sineComponent = torch.sin(inputScaled)
        combinedComponents = torch.stack((cosineComponent, sineComponent), dim=1).reshape(input.shape[0], -1)
        output = self.linear(combinedComponents)
        output = output.reshape(outputShape)
        return output

class TanhLayer(nn.Module):
    '''
    TanhLayer is a custom PyTorch module that applies the Hyperbolic Tangent activation function to the output of a linear transformation.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, the linear transformation includes a bias term. Default is True.
        is_first (bool, optional): If it is the first layer, we initialize the weights differently. Default is False.
        omega_0 (float, optional): Frequency scaling factor for the sinusoidal activation. Default is 30.
        scale (float, optional): Scaling factor for the output of the sine activation. Default is 10.0.
        init_weights (bool, optional): If True, initializes the layer's weights according to the SIREN paper. Default is True.
    '''
    def __init__(self, in_features, out_features, latent_dim, bias=True,
                 is_first=False, omega_0=30, scale=10.0,init_weights=True):
        super().__init__()
        self.latent_dimensions = latent_dim
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
        self.shift_modulation = nn.Linear(latent_dim, out_features)
        self.scale_modulation = nn.Linear(latent_dim, out_features)
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
    
    def forward(self, input, modulation_vector=None):
        sigmoid = nn.Sigmoid()
        shift_modulation_vector = self.shift_modulation(modulation_vector)
        # scale_modulation_vector = self.scale_modulation(modulation_vector)
        linear_output = self.linear(input)
        # linear_output *= scale_modulation_vector.unsqueeze(-2)
        linear_output += shift_modulation_vector.unsqueeze(-2)

        tanh_output = torch.tanh(self.omega_0 * linear_output)
        sigmoid_output = sigmoid(linear_output)
        output = (linear_output + tanh_output) * sigmoid_output
        return output
        
# Modulation Versions of FKAN Layers and FKAN_INR
class FKANLayer_Modulated(nn.Module):
    def __init__(self, in_features, out_features, latent_dim, fourier_bands):
        super(FKANLayer_Modulated, self).__init__()
        self.fkan = FourierKANLayer_TorchLike(in_features, out_features, fourier_bands)
        self.norm = nn.LayerNorm(out_features)

        self.scaleModulation = nn.Linear(latent_dim, out_features)
        self.shiftModulation = nn.Linear(latent_dim, out_features)

    def forward(self, x, latent_vector):
        # compute FourierKANLayer output
        x = self.fkan(x)
        x = self.norm(x)

        #compute modulations
        scale = self.scaleModulation(latent_vector)
        shift = self.shiftModulation(latent_vector)

        while scale.dim() < x.dim():
            scale = scale.unsqueeze(-2)  
            shift = shift.unsqueeze(-2)

        # compute different modulated outputs

        output_unmodulated = x
        # output_scale_shift_modulated = (x * scale) + shift
        # output_scale_modulated = x * scale
        # output_shift_modulated = x + shift

        return output_unmodulated

class FKAN_INR_Modulated(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, latent_dimensions, fourier_bands):
        super(FKAN_INR_Modulated, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.fkan=FKANLayer_Modulated(in_features, hidden_features,self.latent_dimensions,fourier_bands)
        self.hid1=TanhLayer(hidden_features,2*hidden_features, self.latent_dimensions)
        self.hid2=TanhLayer(2*hidden_features,2*hidden_features, self.latent_dimensions)
        self.hid3=TanhLayer(2*hidden_features,2*hidden_features, self.latent_dimensions)
        self.hid4=TanhLayer(2*hidden_features,4*hidden_features, self.latent_dimensions)
        self.out=nn.Linear(4*hidden_features,out_features)
        with torch.no_grad():
            const = np.sqrt(6/hidden_features)/30
            self.out.weight.uniform_(-const, const)
                
    def forward(self, coords, fkan_modulation_latent_vector):
        x=self.fkan(coords, fkan_modulation_latent_vector)
        y1=self.hid1(x, fkan_modulation_latent_vector)
        y2=self.hid2(y1, fkan_modulation_latent_vector)
        y3=self.hid3(y2, fkan_modulation_latent_vector)
        y4=self.hid4(y3, fkan_modulation_latent_vector)
        y4=self.out(y4)
        return y4