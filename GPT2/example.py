import torch
import tiktoken

from model import GPT2, Config


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using greedy decoding (argmax sampling).
    
    Args:
        model: The GPT2 model
        idx: (B, T) array of indices in the current context
        max_new_tokens: Number of new tokens to generate
        context_size: Maximum context length the model supports
    
    Returns:
        idx: (B, T + max_new_tokens) array with generated tokens appended
    """
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


if __name__ == "__main__":
    # Initialize model and tokenizer
    config = Config()
    model = GPT2(config)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load trained model
    model.load_state_dict(torch.load("gpt_custom.pt"))
    model.eval()

    # Starting prompt
    start_context = "The bluefooted booby is a bird found primarily in the"

    # Encode input
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # Display input
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # Generate text
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=config.context_length
    )
    
    # Decode output
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    # Display output
    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)