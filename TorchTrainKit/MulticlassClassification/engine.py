# from src.MulticlassClassification._uitlity import History
import torch
from torch  import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchinfo import summary
import torchmetrics as tm
from torchmetrics.classification import (Accuracy,
                                        Precision,
                                        Recall,
                                        F1Score,
                                        ConfusionMatrix)
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
import requests
from io import BytesIO


class Engine:
    def __init__(self,
                 model=None,
                 train_loader=None,
                 val_loader=None,
                 optimizer=None,
                 lr_scheduler=None,
                 lr_step_type='epoch step', # or can be 'batch step'
                 loss_function=None,
                 test_loader=None,
                 device='cpu',
                 patience=None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader if test_loader else None
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.step_type = lr_step_type
        self.loss_function = loss_function
        self.device = device
        self.history = {}
        self.transforms = None
        self.patience = patience
        self.__dumm_shp = None
        self.__class_names = None
        self.__cls2idx = None
        self.__is_staged = False
        self.__run_logs = []


    def __str__(self):
        return f"""
Engine Class
------------
Model : {self.model.__class__.__name__ if self.model else None}
Train Loader : {type(self.train_loader).__name__ if self.train_loader else None}
Val Loader : {type(self.val_loader).__name__ if self.val_loader else None}
Test Loader : {type(self.test_loader).__name__ if self.test_loader else None}
Optimizer : {self.optimizer.__class__.__name__ if self.optimizer else None}
Loss Function : {self.loss_function.__class__.__name__ if self.loss_function else None}
Patience : {self.patience}
Device : {self.device}
History : {type(self.history).__name__} ({len(self.history.keys())} Key : Value)
Transformations : {self.transforms}
Input Shape : {self.__dumm_shp}
Class Names : {self.__class_names}
Class to Index : {self.__cls2idx}
Is Staged : {self.__is_staged}
Run Logs : {len(self.__run_logs)} Registered Logs
------------
        """

    @staticmethod
    def set_device(device:str):
        """
        checks and sets the best device automatically
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead")
            device = "cpu"
        elif device != "cuda" and device != "cpu":
            print("Device not supported, using CPU instead")
            device = "cpu"
        print(f"[INFO] Device set to {device.upper()}.")
        return device

    def stage(self):
        """
        Stages the engine before training.
        
        -   Model Training
            Must be called if the model is being trained for the first time.
        
        -   Model Loading
            If the model weight is  being loaded, not need to call this method.
            It will be automativally staged, only if the model was originally
            trained using this engine.
        """
        if self.__is_staged:
            print("[INFO] : Engine is already Staged")
            return self

        print("[INFO] Staging ...")
        assert isinstance(self.model, nn.Module), "Model has to be of type nn.Module"
        print("[INFO] Model Passed.")

        assert isinstance(self.train_loader, DataLoader), "For Training, Train Loader has to be of type DataLoader"
        assert self.train_loader is not None , "For Training, Train Loader can not be None"
        print("[INFO] Train Loader Passed.")

        assert self.val_loader is not None , "For Training, Val Loader can not be None"
        assert isinstance(self.val_loader, DataLoader) or not self.val_loader, "For Training, Val Loader has to be of type DataLoader"
        print("[INFO] Validation Loader Passed.")

        if self.train_loader and self.val_loader:
            x, _  = next(iter(self.train_loader))
            self.__class_names = self.train_loader.dataset.classes
            self.__cls2idx = self.train_loader.dataset.class_to_idx
            self.__dumm_shp = x.shape
            self.transforms = self.val_loader.dataset.transforms
            del x, _

        if self.test_loader:
            assert isinstance(self.test_loader, DataLoader), "Test Loader has to be of type DataLoader"
            print("[INFO] Test Loader Passed.")

        assert isinstance(self.loss_function, nn.Module), "Loss Function has to be of type nn.Module"
        print("[INFO] Loss Function Passed.")

        assert isinstance(self.optimizer, Optimizer), "Optimizer has to be of type torch.optim.Optimizer"
        print("[INFO] Optimizer Passed.")

        self.device = self.set_device(self.device)
        print(f"[INFO] Setting model to target device")
        self.model.to(self.device)
        print(f"[INFO] Model is on device: {next(self.model.parameters()).device}")
        print("[INFO] Staged. Engine is ready to train the model.")
        self.__is_staged = True
        return self

    def view_summary(self):
        """
        Shows summary of the model architecture, and the Engine internals
        """
        if not self.__is_staged:
            raise Exception("Engine is not staged, call stage method first")

        if self.model:
            print(summary(model=self.model,
                    input_size=self.__dumm_shp,
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    col_width=20,
                    row_settings=["var_names"],
                    depth=3))
        print(f"""
-------------- Engine Summary --------------
Model : {self.model.__class__.__name__ if self.model else None}
Loss Function : {self.loss_function.__class__.__name__ if self.loss_function else None}
Optimizer : {self.optimizer.__class__.__name__ if self.optimizer else None}
EarlyStopping : {f"Active (Patience {self.patience})" if self.patience is not None and self.patience>0 else "Deactive"}

Image Shape (C, H, W) : {tuple(list(self.__dumm_shp[-3:])) if self.__dumm_shp else [None, None, None]}
Batch Size : {self.__dumm_shp[0] if self.__dumm_shp else 0}
Number of Classes : {len(self.__class_names) if self.__class_names else 0}
CLass Names : {self.__class_names if self.__class_names else None}
Class to index : {self.__cls2idx if self.__cls2idx else None}

Train Loader Size : {len(self.train_loader) if self.train_loader else 0}
Val Loader Size : {len(self.val_loader) if self.val_loader else 0}
Test Loader Size : {len(self.test_loader) if self.test_loader else 0}
Train Transforms : {self.train_loader.dataset.transforms if self.train_loader else None}
Val Transforms : {self.val_loader.dataset.transforms if self.val_loader else None}
Test Transforms : {self.test_loader.dataset.transforms if self.test_loader else None}

Device : {self.device}
------------------- END --------------------
        """
)


    def model_metrics(self):
        """
        collection of models metrics for testing evaluation
        """
        class_len = len(self.__class_names)

        metric_collections = tm.MetricCollection({
            "accuracy": Accuracy(task='multiclass', num_classes=class_len),
            "precision_macro": Precision(task='multiclass', average='macro', num_classes=class_len),
            "recall_macro": Recall(task='multiclass', average='macro', num_classes=class_len),
            "f1_macro": F1Score(task='multiclass', average='macro', num_classes=class_len),
            "f1_weighted": F1Score(task="multiclass", num_classes=class_len, average="weighted"),
            "confusion_matrix": ConfusionMatrix(task='multiclass', num_classes=class_len)
        }).to(self.device)

        return metric_collections


    def _measure_accuracy(self, y_logit, y_true):
        """
        measures accuracy given the y_logit and y_true
        """
        pred_class = torch.argmax(torch.softmax(y_logit, dim=1), dim=1)
        acc = (pred_class == y_true).sum().item() / len(y_logit)
        return acc

    def _train_step(self):
        """
        Training Step of the train loop
        """
        end_size = len(self.train_loader)
        train_loss = 0
        train_acc = 0

        self.model.train()
        for batch_size, (X, y) in tqdm(enumerate(self.train_loader), total=end_size, desc="Running Train Steps"):
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            loss = self.loss_function(logits, y)

            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_acc += self._measure_accuracy(logits, y)

        train_loss = train_loss/len(self.train_loader) # per batch
        train_acc = train_acc/len(self.train_loader)

        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)

        return train_loss, train_acc

    def _val_step(self):
        """
        Validation Step of the train loop
        """
        end_size = len(self.val_loader)
        val_loss = 0
        val_acc = 0
        self.model.eval()
        with torch.inference_mode():
            for batch_size, (X, y) in tqdm(enumerate(self.val_loader), total=end_size, desc="Running Val Steps"):
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.loss_function(logits, y)
                val_loss += loss.item()
                val_acc += self._measure_accuracy(logits, y)

        val_loss = val_loss/len(self.val_loader) # per batch
        val_acc = val_acc/len(self.val_loader)

        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        return val_loss, val_acc


    def train(self, epochs, patience=0,min_delta=0.0, restore_best_weights = True):

        """
        Function trains the provided model on the provided train loader and evaluates on the
        provided validation data loader
        Parameters
        ----------
        epochs : int
            Number of epochs to train the model
        min_delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement, by default 0.0
        restore_best_weights : bool, optional
            Whether to restore model weights from the epoch with the best value of the monitored quantity.
        """
        if not self.__is_staged:
            raise Exception("Engine is not staged, call stage method first")

        import copy
        if len(self.history.keys()) == 0:
            self.history = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": [],
                "epochs": [],
                "lr":[]
            }
        best_val_loss = float("inf")
        patience_ctr = 0
        current_lr = 0
        best_model_weights = None

        for epc in tqdm(range(epochs), desc="Training"):
            train_loss, train_acc = self._train_step()
            val_loss, val_acc = self._val_step()

            if self.scheduler is not None and self.step_type == 'epoch step':
                self.scheduler.step()
                lr_val = self.scheduler.get_last_lr()
                current_lr = lr_val[0] if isinstance(lr_val, list) else lr_val
                self.history['lr'].append(current_lr)
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['lr'].append(current_lr)

            self.history['epochs'].append(epc)
            log = f"\nEpoch {epc + 1}\nTrain Loss: {train_loss:.5f} | Train Acc: {(train_acc*100):.3f}%\nVal Loss: {val_loss:.4f} | Val Acc: {(val_acc*100):.4f}%\nCurrent Lr : {current_lr:.5f}"
            print(log)
            self.__run_logs.append(log)
            if not patience <= 0:
                if val_loss < (best_val_loss - min_delta):
                    log = f"[INFO] Validation Loss improved from {best_val_loss:.4f} to {val_loss:.4f}"
                    print(log)
                    self.__run_logs.append(log)
                    best_val_loss = val_loss
                    patience_ctr = 0
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    patience_ctr += 1
                    log = f"[INFO] Early stopping counter: {patience_ctr}/{patience}"
                    print(log)
                    self.__run_logs.append(log)
                    if patience_ctr >= patience:
                        log = "[INFO] Early stopping triggered"
                        print(log)
                        self.__run_logs.append(log)
                        if restore_best_weights and best_model_weights is not None:
                            self.model.load_state_dict(best_model_weights)
                            log = "[INFO] Best model weights restored"
                            print(log)
                            self.__run_logs.append(log)
                        break

        print("\n[INFO] Access the trained model and its history using")
        print("""
engine = Engine(...)
trained_model = engine.model
history = engine.history
        """)
        return self

    def EvalOnTest(self):
        """
        Puts model on inference mode and calculates 
        Overall accuracy, overall loss, precision (Macro), recall (Macro),
        F1-Score (Macro), F1-Score (Weighted), Confusion Matrix.
        ** Note. This only works if test_loader is provided to Engine
        during initialization
        """
        if not self.__is_staged:
            raise Exception("Engine is not staged, call stage method first")
        if self.test_loader is None:
            raise Exception("Test Loader is not provided")

        collections = self.model_metrics()
        test_loss = 0
        self.model.eval()
        with torch.inference_mode():
            for b, (X, y) in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Running Test Steps"):
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                loss = self.loss_function(logits, y)
                test_loss += loss.item()
                collections.update(logits, y)

        test_loss = test_loss/len(self.test_loader)
        computed_metrics = collections.compute()
        conf_matrix = computed_metrics['confusion_matrix'].cpu().numpy()
        print(f"""\n
==================== Test Set Evaluation ====================
Test Loss        : {test_loss:.4f}
Overall Accuracy : {computed_metrics['accuracy'].item() * 100:.2f}%
Precision (Macro): {computed_metrics['precision_macro'].item():.4f}
Recall (Macro)   : {computed_metrics['recall_macro'].item():.4f}
F1-Score (Macro) : {computed_metrics['f1_macro'].item():.4f}
F1-Score (Weight): {computed_metrics['f1_weighted'].item():.4f}
=============================================================
        """)
        collections.reset()
        metrics = {
            "test_loss": test_loss,
            "accuracy": computed_metrics["accuracy"].item(),
            "precision_macro": computed_metrics["precision_macro"].item(),
            "recall_macro": computed_metrics["recall_macro"].item(),
            "f1_macro": computed_metrics["f1_macro"].item(),
            "f1_weighted": computed_metrics["f1_weighted"].item(),
            "confusion_matrix": conf_matrix
        }
        # We update the history to get all the plots in the plot_history method
        self.history.update(metrics)
        return metrics

    def plot_history(self):

        """
        Plots History data
        """

        if len(self.history.keys()) == 0:
            raise Exception("No history found, call train method first")

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        ax1, ax2 = axs[0, 0], axs[0, 1]
        ax3, ax4 = axs[1, 0], axs[1, 1]

        ax1.set_title("Training and Validation Accuracy Graph")
        ax1.plot(self.history["epochs"], self.history["train_acc"], label="Train")
        ax1.plot(self.history["epochs"], self.history["val_acc"], label="Validation")
        ax1.set_xlabel('Epochs'); ax1.set_ylabel('Accuracy (%)')
        ax1.legend(); ax1.grid(True)

        ax2.set_title("Training and Validation Loss Graph")
        ax2.plot(self.history["epochs"], self.history["train_loss"], label="Train")
        ax2.plot(self.history["epochs"], self.history["val_loss"], label="Validation")
        ax2.set_xlabel('Epochs'); ax2.set_ylabel('Loss')
        ax2.legend(); ax2.grid(True)

        required_keys = ('test_loss', 'accuracy', 'precision_macro', 'recall_macro',
                        'f1_macro', 'f1_weighted', 'confusion_matrix')
        if all(key in self.history for key in required_keys):
            ax3.set_title('Confusion Matrix')
            sns.heatmap(self.history["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
                        ax=ax3, cbar=False,
                        xticklabels=self.__class_names, yticklabels=self.__class_names)
            ax3.set_xlabel('Predicted Labels'); ax3.set_ylabel('True Labels')

            ax4.axis('off')
            metrics_data = (
                f"Test Loss        : {self.history['test_loss']:.4f}\n\n"
                f"Overall Accuracy : {self.history['accuracy'] * 100:.2f}%\n\n"
                f"Precision (Macro): {self.history['precision_macro']:.4f}\n\n"
                f"Recall (Macro)   : {self.history['recall_macro']:.4f}\n\n"
                f"F1-Score (Macro) : {self.history['f1_macro']:.4f}\n\n"
                f"F1-Score (Weight): {self.history['f1_weighted']:.4f}"
            )
            ax4.text(0.5, 0.5, metrics_data, ha='center', va='center', fontsize=15)
            ax4.set_title('Final Evaluation Metrics', pad=15)
        else:
            ax3.axis('off')
            ax4.axis('off')

        plt.tight_layout()
        plt.show()
        return self

    def show_logs(self):
        if self.__run_logs is not None:
            for log in self.__run_logs:
                print(log)
            return
        print("[INFO] No Logs to show")
        return
    
    def save(self, file_path="Model.pt"):

        """
        Saves the trained model weights
        Parameters:
            - file_path : Path where the model will be saved
        """

        if not file_path.endswith((".pt", ".pth")):
            file_path += ".pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_class": self.optimizer.__class__,
            "optimizer_defaults": self.optimizer.defaults, # this will save the weight decay, n=betas and other parameters of the optimizer
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_function_class":  self.loss_function.__class__,
            "loss_function_state_dict": self.loss_function.state_dict() if hasattr(self.loss_function, "state_dict") else None,

            "history": self.history,
            "transforms": self.transforms,
            "class_names": self.__class_names,
            "cls2idx": self.__cls2idx,
            "input_shape": self.__dumm_shp,
            "is_staged": self.__is_staged
        }

        torch.save(checkpoint, file_path)
        print(f"[INFO] Checkpoint successfully saved to '{file_path}'")
        return self

    def load(self, file_path="Model.pt"):
        """
        Loads a pretrained weight and stages the engine. 
        if the model was trained using this engine, 
        after loading the weights, training can be resumed.

        **Note: For loading a weight, the model architecture must be initialized first
    
        Parameters:
            - file_path : Path to the model file .pt
        """
        if not file_path.endswith((".pt", ".pth")):
            file_path += ".pt"
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        print("[INFO] Model Loaded")
        optimizer_class = checkpoint.get("optimizer_class")
        optimizer_defaults = checkpoint.get("optimizer_defaults", {})
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_defaults)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[INFO] Optimizer state dict loaded")

        self.loss_function = checkpoint.get("loss_function_class", None)()
        if checkpoint.get("loss_function_state_dict") is not None and hasattr(self.loss_function, "load_state_dict"):
            self.loss_function.load_state_dict(checkpoint["loss_function_state_dict"])
            print(f"[INFO] Loss Function state dict loaded")

        # restoring training history and other metadatas
        self.history = checkpoint.get("history", None)
        self.transforms = checkpoint.get("transforms", None)
        self.__class_names = checkpoint.get("class_names", None)
        self.__cls2idx = checkpoint.get("cls2idx", None)
        self.__dumm_shp = checkpoint.get("input_shape", None)
        self.__is_staged = checkpoint.get("is_staged", True)

        print(f"[INFO] Checkpoint successfully loaded from '{file_path}' onto {self.device.upper()}")
        return self

    def predict_image(self, image_source, actual_class=None, show_image=True):
        """
        Runs inference on a single image and optionally displays it with a title:
        'Actual Class Name | Predicted Class Name | Probability'

        Parameters
        ----------
        image_source : str
            Local file path or direct image URL (http/https).
        actual_class : str, optional
            Ground-truth class name, if known. If omitted, the title only
            shows the predicted class and probability.
        show_image : bool, optional
            Whether to display the image with matplotlib. Default True.

        Returns
        -------
        dict with keys: "predicted_class", "probability", "actual_class"
        """
        if not self.__is_staged:
            raise Exception("Engine is not staged, call stage method first")
        if self.transforms is None:
            raise Exception("No transforms found on the engine, stage or load first")
        if self.__class_names is None:
            raise Exception("No class names found on the engine, stage or load first")

        # Load image from a URL or a local path
        if isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(image_source).convert("RGB")

        # Transform -> batch of 1 -> device
        transform_fn = self.transforms
        if hasattr(transform_fn, "transform"):   # unwrap StandardTransform if present
            transform_fn = transform_fn.transform
        input_tensor = transform_fn(img).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.inference_mode():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        predicted_class = self.__class_names[pred_idx.item()]
        probability = confidence.item()

        if show_image:
            title_parts = []
            if actual_class is not None:
                title_parts.append(f"Actual: {actual_class}")
            title_parts.append(f"Predicted: {predicted_class}")
            title_parts.append(f"Probability: {probability:.2%}")
            title = " | ".join(title_parts)

            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(title, fontsize=12)
            plt.axis("off")
            plt.show()

        return {
            "predicted_class": predicted_class,
            "probability": probability,
            "actual_class": actual_class
        }





if __name__ == '__main__':
    e = Engine()