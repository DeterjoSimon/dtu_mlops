import sys
import argparse

from tqdm import tqdm
import torch
import numpy as np
from torch import optim, nn
from data import mnist
from model import MyAwesomeModel
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        self.criterion = nn.NLLLoss()
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(f'Running training with learning rate:{args}%')

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        optimizer = optim.Adam(model.parameters(), lr=float(vars(args)['lr']))
        epochs = 50
        train_losses = []
        model.train()
        for e in range(epochs):
            running_loss = 0
            for images, labels in train_set:
                
                log_ps = model.forward(images)
                optimizer.zero_grad()
                loss = self.criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
            else:
                train_losses.append(running_loss/len(train_set)) 
                print("Training loss: {}".format(running_loss/len(train_set)))

        # Plotting training step vs. training loss
        plt.plot(np.arange(0,len(train_losses)),train_losses)
        plt.xlabel("training step")
        plt.ylabel("training loss")
        plt.show()

        # Saving model
        torch.save(model.state_dict(), 'checkpoint.pth')


        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args.load_model_from)
        if args.load_model_from:
            state_dict = torch.load(args.load_model_from)
            model = MyAwesomeModel()
            model.load_state_dict(state_dict)
        # TODO: Implement evaluation logic here

        
        _, test_set = mnist()
        steps = 0
        model.eval()
        with torch.no_grad():
            accuracy = 0
            test_loss = 0
            for images, labels in test_set:
                
                steps += 1
                log_ps = model(images)
                loss = self.criterion(log_ps, labels)
                test_loss += loss.item()
                if steps % 20 == 0:
                    print("Test Loss: {:.3f}..".format(test_loss))

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        top_p, top_class = torch.exp(log_ps).topk(1, dim=1)
        # Class with highest probability is our predicted class, compare with true label
        equality = top_class == labels.view(*top_class.shape)
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy = torch.mean(equality.type(torch.FloatTensor))

        print("Test Accuracy: {:.3f}..".format(accuracy.item()*100))
        



if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    