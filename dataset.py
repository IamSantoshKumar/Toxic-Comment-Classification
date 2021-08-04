import torch
class ToxicDataset:
    def __init__(self, reviews, targets):

       self.reviews = reviews
       self.target = targets
    def __len__(self):
       # returns length of the dataset
       return len(self.reviews)
    def __getitem__(self, item):
       # for any given item, which is an int,
       # return review and targets as torch tensor
       # item is the index of the item in concern
       review = self.reviews[item, :]
       target = self.target[item]
       return {
              "review": torch.tensor(review, dtype=torch.long),
              "target": torch.tensor(target, dtype=torch.float)
       }