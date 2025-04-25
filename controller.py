import abc
import torch
import numpy as np

from diffusers import StableDiffusionPipeline
from diffusers.utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

"""
This class acts as an abstract base for controlling attention layers during the model's operation.
"""
class AttentionControl(abc.ABC):

    # This method appears to serve as a placeholder callback for each step, currently returning the input x_t unchanged.
    # It could be overridden in subclasses for specific behavior after each processing step.
    def step_callback(self, x_t):
        return x_t

    # Another method intended as a hook for actions to be performed between steps of the model's operation.
    # It does nothing in the base class and can be overridden.
    def between_steps(self):
        return

    # A property that returns the number of unconditional attention layers.
    # This number depends on whether the model is operating in a "LOW_RESOURCE" mode, which seems to be a flag not defined within this snippet.
    @property
    def num_uncond_att_layers(self):
        return 0

    # An abstract method that must be implemented by subclasses. It's designed to process the attention matrix attn,
    # with additional context provided by is_cross (indicating if the attention is cross-layer) and place_in_unet (the location in the U-Net architecture).
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    # The main method that wraps the forward method, providing additional management of attention layers and step tracking.
    # It adjusts the attention matrix depending on the current attention layer and other factors, and resets layer tracking after completing all layers.
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    # Resets internal counters for steps and layers.
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    # Initializes instance variables for tracking steps and layers.
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


"""
This subclass of AttentionControl is specifically for storing and aggregating attention matrices over multiple steps.
"""
class AttentionStore(AttentionControl):

    # A static method that returns a structured empty storage for different types of attention matrices categorized by their positions and types (cross or self).
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    # Overrides the forward method to store the attention matrix in step_store based on its characteristics (size and type).
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] == 64 ** 2 and key == "down_cross" and self.cur_step == 49:  # avoid memory overhead <= 32 ** 2 changed to == 64 ** 2
            reshaped_attn = attn.view(-1, 8, attn.shape[1], attn.shape[2])
            reshaped_attn = reshaped_attn.mean(dim=1)
            self.step_store[key].append(reshaped_attn) 
        return attn

    # Manages the transition between steps, aggregating stored attention matrices from step_store into a longer-term attention_store.
    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    # Calculates the average of stored attention matrices over all steps,
    def get_average_attention(self):
        # self.attention_store["down_cross"] is a list of 2 tensors of shape [4096, 77], [4096, 77].
        average_attention = {key: [item for item in self.attention_store[key]] for key in self.attention_store} 
        return average_attention 

    # Overrides the base class reset to clear both the step-specific and overall attention stores.
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    # Initializes both the step-specific and overall attention stores upon object creation.
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}