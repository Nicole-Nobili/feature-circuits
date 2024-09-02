import torch as t
from nnsight import LanguageModel


model = LanguageModel('EleutherAI/pythia-70m-deduped', device_map="cuda:0", dispatch=True)

with model.trace("The eiffel tower is in Paris"), t.no_grad():
    # Access and process the output of the embedding layer
    embed_output = model.gpt_neox.embed_in.output
    print("Embedding output shape:", embed_output.shape)

    # Access and process the output of each layer
    for i, layer in enumerate(model.gpt_neox.layers):
        # Attention output
        attn_output = layer.attention.output
        print(f"Layer {i} attention output shape:", attn_output.shape)
        
        # MLP output
        mlp_output = layer.mlp.output
        print(f"Layer {i} MLP output shape:", mlp_output.shape)
        
        # Residual output (layer output)
        residual_output = layer.output
        print(f"Layer {i} residual output shape:", residual_output.shape)

    # Access the final output
    final_output = model.embed_out.output
    print("Final output shape:", final_output.shape)

    # Example: Calculate mean activation across all layers
    mean_activation = t.mean(t.stack([layer.output for layer in model.gpt_neox.layers]))
    print("Mean activation across all layers:", mean_activation.item())
