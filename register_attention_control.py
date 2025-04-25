import torch
import numpy as np

from diffusers import StableDiffusionPipeline
from diffusers.utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


"""
This function modifies the attention mechanisms of the U-Net in the Stable Diffusion model,
allowing for custom control over how attention is computed and applied.
The modifications are applied using a custom controller that manipulates attention weights.
This is particularly useful for tasks like targeted image generation or modifying specific features in generated images.
"""

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        """
        Custom forward functions for attention layers.

        The function ca_forward is a higher-order function that modifies the forward method of attention layers within the U-Net.
        It is designed to be a factory function, which means it creates and returns another function (forward).
        This nested forward function is intended to be used as a replacement for the existing forward method of attention layers in the U-Net model.
        Therefore, the ca_forward function needs access to the layer it is modifying.

        Input (self): represents an instance of an attention layer in the U-Net model.
        The function is written to assume that it will be attached to or called on an attention layer instance,
        where self will automatically refer to that layer instance.

        Input (place_in_unet): specifies its position in the U-Net architecture (e.g., "down", "up", "mid").
        """
        # Checks if to_out (the output transformation module of the attention layer) is a list of modules (i.e., ModuleList).
        # If so, it selects the first module for simplicity. Otherwise, it uses the module directly.
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            """
            Redefines the forward computation for attention layers

            Input (hidden_states): the activations (or features) from a previous layer that are input into the attention layer. It is used to compute the 'query' for cross-attention.
            Input (encoder_hidden_states): This parameter is used in the case of cross-attention, where the attention mechanism needs to consider another set of activations different from the hidden_states.
                                           These states typically come from a different part of the model, such as an encoder in a generative model or a different processing stream.
                                           If encoder_hidden_states is provided, it is used to compute the 'key' and 'value' for the attention, linking different parts of the input data.
            Input (attention_mask): It is used to specify which elements in the input should not be attended to (i.e., ignored during the computation of attention scores).
                                    This is typically used to handle variable-length sequences in batch processing or to mask out certain parts of the input for specific reasons.
                                    The mask influences the computation of attention scores by zeroing out masked elements.
            Input (temb): This stands for "temporal embedding" and is often used in generative models to add information about specific attributes or conditions under which the model operates.
                          For example, in image generation, temb could include information about the style or specific characteristics desired in the output.
                          It can be used to modulate the features before they are processed by the attention mechanism.
            """
            # Determines if the attention operation is cross-attention
            is_cross = encoder_hidden_states is not None # shape always [2,77,768]

            # Saves the input hidden_states as residual for possible residual connections later.
            residual = hidden_states # shape [2,4096,320]*2, [2,1024,640]*2, [2,256,1280]*2, [2,64,1280], up*3

            # Applies spatial normalization to the `hidden_states`, if configured in the layer, using `temb`.
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            # Stores the number of dimensions of `hidden_states` to handle reshaping accurately.
            input_ndim = hidden_states.ndim

            # Reshapes hidden_states from a 4D tensor (common in image processing) to a 3D tensor compatible with attention mechanisms,
            # by flattening the spatial dimensions and transposing channels and spatial dimensions.
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            # Determines `batch_size` and `sequence_length` based on the shape of `hidden_states` or `encoder_hidden_states`.
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            # Prepares the attention_mask for the current batch and sequence configuration.
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size) # None

            # Applies group normalization if configured, transposing as necessary for dimensionality alignment.
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            # Computes the query vector from hidden_states, and prepares encoder_hidden_states by normalizing if cross-attention is normalized (norm_cross).
            query = self.to_q(hidden_states) # [2,4096,320]*2, [2,1024,640]*2, [2,256,1280]*2, [2,64,1280], up*3

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            # Computes key and value vectors from encoder_hidden_states.
            key = self.to_k(encoder_hidden_states) # [2,77,320]*2, [2,77,640]*2, [2,77,1280]*2, [2,77,1280], up*3
            value = self.to_v(encoder_hidden_states) # [2,77,320]*2, [2,77,640]*2, [2,77,1280]*2, [2,77,1280], up*3

            # Reshapes query, key, and value for batch processing.
            query = self.head_to_batch_dim(query) # [16,4096,40]*2, [16,1024,80]*2, [16,256,160]*2, [16,64,160], up*3
            key = self.head_to_batch_dim(key) # [16,77,40]*2, [16,77,80]*2, [16,77,160]*2, [16,77,160], up*3
            value = self.head_to_batch_dim(value) # [16,77,40]*2, [16,77,80]*2, [16,77,160]*2, [16,77,160], up*3

            # Computes initial attention probabilities
            attention_probs = self.get_attention_scores(query, key, attention_mask) # [16,4096,77]*2, [16,1024,77]*2, [16,256,77]*2, [16,64,77], up*3

            # Then, modifies them using the controller based on whether the attention is cross or self and its position (up, down, mid) in the U-Net.
            attention_probs = controller(attention_probs, is_cross, place_in_unet) # [16,4096,77]*2, [16,1024,77]*2, [16,256,77]*2, [16,64,77], up*3

            # Applies the attention by matrix-multiplying the attention probabilities with value, then reshapes back to the original dimensionality.
            hidden_states = torch.bmm(attention_probs, value) # [16,4096,40]*2, [16,1024,80]*2, [16,256,160]*2, [16,64,160], up*3
            hidden_states = self.batch_to_head_dim(hidden_states) # [2,4096,320]*2, [2,1024,640]*2, [2,256,1280]*2, [2,64,1280], up*3

            # Transforms the `hidden_states` through the output module (to_out).
            hidden_states = to_out(hidden_states) # linear proj # [2,4096,320]*2, [2,1024,640]*2, [2,256,1280]*2, [2,64,1280], up*3

            # If the original input was 4D, reshapes hidden_states back to this format.
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            # Adds the residual back to hidden_states if the layer uses residual connections.
            if self.residual_connection:
                hidden_states = hidden_states + residual

            # Scales down hidden_states by a predefined factor to stabilize outputs.
            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward

    class DummyController:
        """
        A fallback class used when no specific controller is provided. It simply returns the attention scores without any modification.
        """

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        """
        This function recursively traverses the model's U-Net architecture, applying the modified ca_forward to each attention layer found.
        - It identifies layers by name to determine their position within the U-Net ("down", "up", "mid").
        - The modified forward method is registered to each attention layer found.
        """
        # If the current layer is an attention layer, modifies its forward method and increments the count of modified layers.
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet) # ca_forward is intended to be used.
                                                           # It sets the forward method of an attention layer (net_) to the function returned by ca_forward.
                                                           # net_ serves as the `self` for `ca_forward`, acting as the specific attention layer being modified.
                                                           # ca_forward customizes the behavior of net_ based on its position within the U-Net structure (indicated by place_in_unet).
            return count + 1

        # If the layer contains sub-layers, recursively applies the modifications to each child.
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    # Execution: applies the recursive modifications through register_recr, tallying the modified layers.
    # Counts and categorizes modifications based on their position in the U-Net ("down", "up", "mid").
    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")

    # Sets the count of modified layers in the controller, which could be used for debugging or logging purposes.
    controller.num_att_layers = cross_att_count