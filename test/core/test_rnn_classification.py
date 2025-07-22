import os
import random
import time
import unittest

import numpy as cpu_np
import torch
import torch.nn as nn
import torch.nn.functional as F


from src.lib.core.lif_ffn import LIFFfn

language_alphabets = {"English": set("abcdefghijklmnopqrstuvwxyz"),
                      "French": set("abcdefghijklmnopqrstuvwxyzàâæçéèêëîïôœùûüÿ"),
                      "German": set("abcdefghijklmnopqrstuvwxyzäöüß"),
                      "Spanish": set("abcdefghijklmnopqrstuvwxyzáéíñóúü"),
                      "Russian": set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")}

all_chars = sorted(set().union(*language_alphabets.values()))
char_to_index = {c: i for i, c in enumerate(all_chars)}
input_size = len(char_to_index)
class_labels = list(language_alphabets.keys())
output_size = len(class_labels)


def encode_text_numpy(text):
    vec = cpu_np.zeros(input_size, dtype=cpu_np.float32)
    for c in text.lower():
        if c in char_to_index:
            vec[char_to_index[c]] += 1
    vec /= (cpu_np.linalg.norm(vec) + 1e-8)
    return vec


def one_hot_numpy(index):
    vec = cpu_np.zeros(output_size, dtype=cpu_np.float32)
    vec[index] = 1.0
    return vec


examples = {lang: [(encode_text_numpy(sent), one_hot_numpy(class_labels.index(lang))) for sent in sents] for lang, sents
            in {"English": ["Hello world", "The cat jumps", "Good morning", "I love coding", "She is reading a book",
                            "It is raining outside", "We are going home", "They play football", "This is my dog",
                            "Can you help me?"],
                "French": ["Bonjour le monde", "Je t'aime", "Il fait beau", "J'adore le chocolat", "Elle lit un livre",
                           "Il pleut dehors", "Nous rentrons à la maison", "Ils jouent au football",
                           "Ceci est mon chien",
                           "Peux-tu m'aider ?", "Très bonne langue."],
                "German": ["Hallo Welt", "Ich liebe dich", "Guten Morgen", "Die Katze springt", "Sie liest ein Buch",
                           "Es regnet draußen", "Wir gehen nach Hause", "Sie spielen Fußball", "Das ist mein Hund",
                           "Kannst du mir helfen?"],
                "Spanish": ["Hola mundo", "Te amo", "Buenos días", "Me gusta el chocolate",
                            "Ella está leyendo un libro",
                            "Está lloviendo afuera", "Vamos a casa", "Juegan al fútbol", "Este es mi perro",
                            "¿Puedes ayudarme?"],
                "Russian": ["Привет мир", "Я тебя люблю", "Доброе утро", "Мне нравится шоколад", "Она читает книгу",
                            "На улице идёт дождь", "Мы идём домой", "Они играют в футбол", "Это моя собака",
                            "Ты можешь мне помочь?"]}.items()}
training_data = [{"inputs": x, "target": y, "label": lang} for lang, pairs in examples.items() for x, y in pairs]
random.seed(42)
random.shuffle(training_data)


class TestTorchDynamicFeedForwardNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        This entire method is rewritten to use PyTorch.
        It instantiates the PyTorch model, loads weights from the old .npz file
        if it exists, or trains a new model using the PyTorch training loop.
        """
        # Set up device
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate the PyTorch model
        cls.net = LIFFfn(
            input_size=input_size,
            output_size=output_size,
            hidden_layers_config=[64],
            dropout_rate=0.1,
            dtype= torch.float32
        ).to(cls.device)
        param_path = 'model_torch_rnn.pth'

        # Try to load a pre-trained PyTorch model first
        if os.path.exists(param_path):
            print("Loading pre-trained PyTorch model...")
            cls.net.load_state_dict(torch.load(param_path))
            return

        old_param_path = 'model_dynamic_stateless.npz'
        if os.path.exists(old_param_path):
            print("Loading and converting pre-trained stateless model...")
            data = cpu_np.load(old_param_path)
            state_dict = cls.net.state_dict()

            # Manually map old numpy weights to new PyTorch state_dict
            state_dict['layers.0.weight'] = torch.tensor(data['W0'].T, device=cls.device)
            state_dict['layers.0.bias'] = torch.tensor(data['b0'], device=cls.device)
            state_dict['layers.1.weight'] = torch.tensor(data['W1'].T, device=cls.device)
            state_dict['layers.1.bias'] = torch.tensor(data['b1'], device=cls.device)
            state_dict['layers.2.weight'] = torch.tensor(data['W2'].T, device=cls.device)
            state_dict['layers.2.bias'] = torch.tensor(data['b2'], device=cls.device)

            cls.net.load_state_dict(state_dict)
            torch.save(cls.net.state_dict(), param_path)
            return

        print("Training PyTorch model from scratch...")
        optimizer = torch.optim.AdamW(cls.net.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        cls.net.train()
        for epoch in range(100):
            total_loss, correct = 0, 0
            for sample in training_data:
                x = torch.tensor(sample["inputs"], device=cls.device).view(1, 1, -1)
                y_true_idx = torch.tensor(cpu_np.argmax(sample["target"]), device=cls.device).view(-1)

                optimizer.zero_grad()
                logits = cls.net(x)

                loss = criterion(logits[:, -1, :], y_true_idx)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(cls.net.parameters(), 5.0)

                optimizer.step()

                if torch.argmax(logits) == y_true_idx:
                    correct += 1
                total_loss += loss.item()

            if epoch % 10 == 0:
                acc = correct / len(training_data)
                print(f"Epoch {epoch} - Loss: {total_loss:.4f} - Accuracy: {acc:.2%}")

        # Save the PyTorch model state
        torch.save(cls.net.state_dict(), param_path)

    def _test_prediction(self, sentence, expected):
        # Set model to eval mode and use torch.no_grad()
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(encode_text_numpy(sentence), device=self.device).view(1, 1, -1)
            logits = self.net(x)
            # Softmax is often part of the loss function, but for inference we apply it manually
            probs = F.softmax(logits, dim=-1)
            prediction_idx = torch.argmax(probs).item()
            prediction_label = class_labels[prediction_idx]

        print(f"Input: {sentence}\nPredicted: {prediction_label}, Expected: {expected}")
        self.assertEqual(prediction_label, expected)

    def test_english(self):
        self._test_prediction("This is my english language from America", "English")

    def test_french(self):
        self._test_prediction("Je t'aime", "French")

    def test_german(self):
        self._test_prediction("Dies ist meine deutsche Sprache aus Deutschland", "German")

    def test_spanish(self):
        self._test_prediction("Ella está leyendo un libro", "Spanish")

    def test_russian(self):
        self._test_prediction("Это мой английский язык из Америки.", "Russian")


if __name__ == '__main__':
    unittest.main()
