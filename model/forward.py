import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from model.mlp import MLP
from model.transformer import TransformerForwardModel
from model import parameters
from tqdm import tqdm


class ForwardModel:
    def __init__(self, input_dim, output_dim):
        self.device = (
            torch.device("cpu")
            if parameters.USE_CPU_ONLY
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        if parameters.MODEL_TYPE == "MLP":
            self.model = MLP(
                linear_layers=parameters.LINEAR,
                dropout=parameters.DROPOUT,
                skip_connection=parameters.SKIP_CONNECTION,
                skip_head=parameters.SKIP_HEAD,
            )
        elif parameters.MODEL_TYPE == "Transformer":
            self.model = TransformerForwardModel(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=parameters.HIDDEN_DIM,
                num_layers=parameters.TRANSFORMER_LAYERS,
                num_heads=parameters.TRANSFORMER_HEADS,
                dropout=parameters.DROPOUT_RATE,
            )
        else:
            raise ValueError(f"Unknown model type: {parameters.MODEL_TYPE}")

        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=parameters.LEARN_RATE,
            weight_decay=parameters.REG_SCALE,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=parameters.LR_DECAY_RATE,
            patience=10,
            verbose=True,
        )

        self.best_loss = float("inf")
        self.model_dir = os.path.join(parameters.output_dir, parameters.DATA_SET)
        os.makedirs(self.model_dir, exist_ok=True)

        print("\n Model Architecture:")
        print(self.model)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total learning parameters: {total_params}\n")

    def train(self, train_loader, val_loader):
        training_losses = []
        validation_losses = []
        no_improvement_counter = 0

        for step in range(parameters.TRAIN_STEP):
            self.model.train()
            total_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(
                f"Step [{step+1}/{parameters.TRAIN_STEP}] - Train Loss: {avg_loss:.6f}"
            )

            if (step + 1) % parameters.EVAL_STEP == 0:
                val_loss = self.evaluate(val_loader)
                print(f"Step [{step+1}] - Validation Loss: {val_loss:.6f}")

                training_losses.append(avg_loss)
                validation_losses.append(val_loss)

                self.scheduler.step(val_loss)  # <-- Updated for ReduceLROnPlateau

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_model()
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                if no_improvement_counter >= parameters.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

            if avg_loss < parameters.STOP_THRESHOLD:
                print("Loss below threshold, stopping early.")
                break

        if parameters.PLOT_TRAINING_CURVE:
            self.plot_training_curve(training_losses, validation_losses)

    def plot_training_curve(self, train_losses, val_losses):
        steps = list(
            range(
                parameters.EVAL_STEP,
                parameters.EVAL_STEP * (len(train_losses) + 1),
                parameters.EVAL_STEP,
            )
        )
        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_losses, label="Training Loss")
        plt.plot(steps, val_losses, label="Validation Loss")
        plt.xlabel("Training Step")
        plt.ylabel("MSE Loss")
        plt.title("Loss vs. Training Step")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self.model_dir, "training_curve.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Training curve saved to: {save_path}")

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def save_model(self):
        model_path = os.path.join(self.model_dir, "best_model.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Best model saved to: {model_path}")

    def load_model(self):
        model_path = os.path.join(self.model_dir, "best_model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, data_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x_batch, _ in tqdm(data_loader, desc="Predicting"):
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch)
                predictions.append(outputs.cpu())
        return torch.cat(predictions, dim=0)
