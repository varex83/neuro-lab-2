import cv2
import numpy as np
import torch
import torch.nn as nn
from mnist_cnn import CNN, transform
import matplotlib.pyplot as plt

class DigitDrawer:
    def __init__(self, model_path=None):
        self.model = CNN()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.width = 280
        self.height = 280
        self.radius = 10
        self.color = (255, 255, 255)
        self.thickness = -1

        cv2.namedWindow('Draw Digit')
        cv2.setMouseCallback('Draw Digit', self.draw_circle)
        self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)
        self.drawing = False

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(self.canvas, (x, y), self.radius, self.color, self.thickness)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def preprocess_image(self, img):
        img = cv2.resize(img, (28, 28))
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def predict_digit(self, img_tensor):
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            return pred.item(), probabilities[0].numpy()

    def run(self):
        print("Draw a digit and press 'p' to predict, 'c' to clear, 'q' to quit")

        while True:
            cv2.imshow('Draw Digit', self.canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros((self.height, self.width), dtype=np.uint8)
            elif key == ord('p'):
                img_tensor = self.preprocess_image(self.canvas)
                prediction, probabilities = self.predict_digit(img_tensor)

                print(f"\nPredicted digit: {prediction}")
                print("\nProbabilities for each digit:")
                for i, prob in enumerate(probabilities):
                    print(f"Digit {i}: {prob:.4f}")

                plt.figure(figsize=(10, 4))
                plt.bar(range(10), probabilities)
                plt.title('Prediction Probabilities')
                plt.xlabel('Digit')
                plt.ylabel('Probability')
                plt.xticks(range(10))
                plt.show()

        cv2.destroyAllWindows()

def save_model(model, path='mnist_model.pth'):
    torch.save(model.state_dict(), path)

def load_model(path='mnist_model.pth'):
    model = CNN()
    model.load_state_dict(torch.load(path))
    return model

if __name__ == '__main__':
    try:
        model = load_model()
        print("Loaded existing model")
    except:
        print("Training new model...")
        from mnist_cnn import train_with_config
        model, _, _, _ = train_with_config('adam', 'cross_entropy')
        save_model(model)
        print("Model trained and saved")

    drawer = DigitDrawer('mnist_model.pth')
    drawer.run() 