# GPT-2 Implementation - Improvements Log

## Upgrades V1.0

1. **Train/Val Split + Early Stopping** 
    - Splits data 80/20, stops training when validation loss plateaus. 

2. **AdamW Optimizer** 
    - Better version of Adam with proper weight decay. 

3. **Learning Rate Scheduling** 
    - Automatically reduces learning rate when training stalls. 

4. **Gradient Clipping** 
    - Caps gradient magnitudes at 1.0 to prevent exploding gradients and training instability.

5. **Temperature Sampling** 
    - Adds controlled randomness instead of always picking the most likely token.

6. **Top-p Sampling** 
    - Only considers most probable tokens that sum to 80% probability. 

7. **SwiGLU Activation** 
    - Replaces GELU in feedforward layers. More expressive and performs better in modern LLMs.