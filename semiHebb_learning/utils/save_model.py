import os
import torch


# save model
def save_model(model, model_save_path):
    prompt = input("Do you want to save the pretrained model? (y/n): ").strip().lower()

    if prompt == 'y':
        # Freeze parameters before saving
        for param in model.parameters():
            param.requires_grad = False

        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # Save the model's state dictionary
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")
    elif prompt == 'n':
        print("Model not saved.")
    else:
        print("Invalid input. Model not saved.")